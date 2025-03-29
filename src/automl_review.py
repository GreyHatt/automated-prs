import os
import json
import time
from github import Github, GithubException
from google.cloud import aiplatform
from dotenv import load_dotenv
from diff_match_patch import diff_match_patch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

class CodeReviewer:
    def __init__(self):
        # Initialize configuration
        self.gh_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY")
        self.event_path = os.getenv("GITHUB_EVENT_PATH")
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.region = os.getenv("GCP_REGION")
        self.endpoint_id = os.getenv("ENDPOINT_ID")
        
        # Initialize models
        try:
            print("Initializing CodeReviewer model...")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise
        
        # Initialize Vertex AI
        try:
            aiplatform.init(project=self.project_id, location=self.region)
            self.endpoint = aiplatform.Endpoint(self.endpoint_id)
        except Exception as e:
            print(f"Vertex AI initialization failed: {str(e)}")
            raise
        
        # Initialize GitHub client
        try:
            self.github = Github(self.gh_token)
            self.repo = self.github.get_repo(self.repo_name)
        except GithubException as e:
            print(f"GitHub API connection failed: {str(e)}")
            raise
        
        # Initialize diff tools
        self.dmp = diff_match_patch()

    def get_pr_details(self):
        """Fetch the PR details from GitHub event data"""
        try:
            with open(self.event_path, 'r') as f:
                event_data = json.load(f)
            pr_number = event_data['number']
            return self.repo.get_pull(pr_number)
        except Exception as e:
            print(f"Failed to get PR details: {str(e)}")
            raise

    def parse_diff(self, diff_text):
        """Parse unified diff to extract changed lines with positions"""
        changes = []
        lines = diff_text.split('\n')
        file_path = None
        current_line = None
        
        for line in lines:
            if line.startswith('+++ b/'):
                file_path = line[6:]
            elif line.startswith('@@ '):
                parts = line.split(' ')
                new_part = parts[2]
                new_start = new_part.split(',')[0][1:]
                try:
                    current_line = int(new_start)
                except ValueError:
                    current_line = 1
            elif line.startswith('+') and not line.startswith('++'):
                if file_path and current_line is not None:
                    changes.append({
                        'file_path': file_path,
                        'line_number': current_line,
                        'content': line[1:]
                    })
                current_line += 1
            elif line.startswith(' '):
                current_line += 1
        
        return changes

    def analyze_code(self, code_snippets):
        """Analyze code snippets using the model"""
        results = []
        for snippet in code_snippets:
            try:
                inputs = self.tokenizer(
                    snippet["content"], 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                )
                outputs = self.model.generate(**inputs)
                suggestion = self.tokenizer.decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                
                if suggestion.strip():
                    results.append({
                        'file_path': snippet['file_path'],
                        'line_number': snippet['line_number'],
                        'suggestions': [suggestion]
                    })
            except Exception as e:
                print(f"Failed to analyze code: {str(e)}")
                continue
        
        return results

    def post_comments(self, pr, analysis_results):
        """Post review comments to GitHub PR with retry logic"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Get the head commit
                commit = self.repo.get_commit(pr.head.sha)
                
                # Create review comments
                review_comments = []
                for result in analysis_results:
                    if result['suggestions']:
                        review_comments.append({
                            'path': result['file_path'],
                            'position': result['line_number'],  # Simplified position
                            'body': "\n".join([f"🔍 **Code Review Suggestion**: {s}" for s in result['suggestions']])
                        })
                        time.sleep(1)  # Rate limiting
                
                if review_comments:
                    pr.create_review(
                        commit=commit,
                        body="Automated code review suggestions",
                        event="COMMENT",
                        comments=review_comments
                    )
                    return True
                return False
            except GithubException as e:
                if attempt == max_attempts - 1:
                    print(f"GitHub API error after {max_attempts} attempts: {str(e)}")
                    raise
                wait_time = (2 ** attempt)  # Exponential backoff
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Failed to post comments: {str(e)}")
                raise

    def run(self):
        """Main execution flow"""
        try:
            pr = self.get_pr_details()
            files = pr.get_files()
            
            all_changes = []
            for file in files:
                if file.filename.endswith(('.py', '.js', '.java', '.go', '.ts', '.cpp', '.h')):
                    if file.patch:
                        changes = self.parse_diff(file.patch)
                        all_changes.extend(changes)
            
            if all_changes:
                print(f"Found {len(all_changes)} changes to analyze")
                analysis_results = self.analyze_code(all_changes)
                if analysis_results:
                    print(f"Posting {len(analysis_results)} suggestions")
                    success = self.post_comments(pr, analysis_results)
                    if success:
                        print("Comments posted successfully")
                    else:
                        print("No comments were posted")
                else:
                    print("No suggestions generated")
            else:
                print("No code changes found to analyze")
        except Exception as e:
            print(f"Error in code review process: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        reviewer = CodeReviewer()
        reviewer.run()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        exit(1)