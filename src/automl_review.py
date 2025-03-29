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
            print(f"Processing PR #{pr_number}")
            return self.repo.get_pull(pr_number)
        except Exception as e:
            print(f"Failed to get PR details: {str(e)}")
            raise

    def parse_diff(self, diff_text):
        """Parse unified diff to extract changed lines with positions"""
        changes = []
        if not diff_text:
            print("Empty diff text received")
            return changes

        lines = diff_text.split('\n')
        file_path = None
        current_line = None
        in_hunk = False
        
        print(f"Parsing diff with {len(lines)} lines")
        
        for line in lines:
            if line.startswith('+++ b/'):
                file_path = line[6:]
                print(f"Found file: {file_path}")
            elif line.startswith('@@ '):
                parts = line.split(' ')
                if len(parts) >= 3:
                    new_part = parts[2]
                    new_start = new_part.split(',')[0][1:]
                    try:
                        current_line = int(new_start)
                        print(f"Found new hunk starting at line {current_line}")
                        in_hunk = True
                    except ValueError:
                        current_line = 1
                        in_hunk = True
            elif in_hunk:
                if line.startswith('+') and not line.startswith('++'):
                    if file_path and current_line is not None:
                        changes.append({
                            'file_path': file_path,
                            'line_number': current_line,
                            'content': line[1:]
                        })
                        print(f"Found added line at {file_path}:{current_line} - {line[:50]}...")
                    current_line += 1
                elif line.startswith(' '):
                    current_line += 1
                elif line.startswith('diff --git'):
                    in_hunk = False
            
        print(f"Found {len(changes)} changes in file {file_path}")
        return changes

    def analyze_code(self, code_snippets):
        """Analyze code snippets using the model"""
        results = []
        print(f"Analyzing {len(code_snippets)} code snippets")
        
        for snippet in code_snippets:
            try:
                if not snippet['content'].strip():
                    continue
                    
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
                    print(f"Generated suggestion for {snippet['file_path']}:{snippet['line_number']}")
            except Exception as e:
                print(f"Failed to analyze code at {snippet['file_path']}:{snippet['line_number']}: {str(e)}")
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
                        comment_body = "🔍 **Code Review Suggestions**:\n" + "\n".join([f"- {s}" for s in result['suggestions']])
                        review_comments.append({
                            'path': result['file_path'],
                            'position': result['line_number'],
                            'body': comment_body
                        })
                        print(f"Prepared comment for {result['file_path']}:{result['line_number']}")
                        time.sleep(1)  # Rate limiting
                
                if review_comments:
                    print(f"Posting {len(review_comments)} comments to GitHub")
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
                wait_time = (2 ** attempt)
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Failed to post comments: {str(e)}")
                raise

    def run(self):
        """Main execution flow"""
        try:
            pr = self.get_pr_details()
            files = list(pr.get_files())
            print(f"Found {len(files)} files in PR")
            
            all_changes = []
            for file in files:
                print(f"\nProcessing file: {file.filename}")
                if not any(file.filename.endswith(ext) for ext in ['.py', '.js', '.java', '.go', '.ts', '.cpp', '.h', '.rb', '.php', '.sh']):
                    print(f"Skipping non-code file: {file.filename}")
                    continue
                
                if file.patch:
                    print(f"Found patch for {file.filename} ({len(file.patch)} chars)")
                    changes = self.parse_diff(file.patch)
                    if changes:
                        all_changes.extend(changes)
                    else:
                        print(f"No changes detected in {file.filename}")
                else:
                    print(f"No patch available for {file.filename} (possibly binary file)")
            
            if all_changes:
                print(f"\nFound {len(all_changes)} changes to analyze")
                analysis_results = self.analyze_code(all_changes)
                if analysis_results:
                    print(f"\nPosting {len(analysis_results)} suggestions")
                    success = self.post_comments(pr, analysis_results)
                    if success:
                        print("Comments posted successfully")
                    else:
                        print("No comments were posted")
                else:
                    print("No suggestions generated by the model")
            else:
                print("\nNo code changes found to analyze. Details:")
                print(f"- Total files in PR: {len(files)}")
                print(f"- Files with patches: {sum(1 for f in files if f.patch)}")
                print(f"- Files without patches: {sum(1 for f in files if not f.patch)}")
                print("Supported file extensions: .py, .js, .java, .go, .ts, .cpp, .h, .rb, .php, .sh")
        except Exception as e:
            print(f"\nError in code review process: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        print("Starting code review process...")
        reviewer = CodeReviewer()
        reviewer.run()
        print("Process completed")
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        exit(1)