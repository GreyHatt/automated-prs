import os
import json
import time
import re
from github import Github, GithubException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

class CodeReviewer:
    def __init__(self):
        # Initialize configuration
        self.gh_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY")
        self.event_path = os.getenv("GITHUB_EVENT_PATH")
        
        # Initialize models
        try:
            print("Initializing CodeReviewer model...")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise
        
        # Initialize GitHub client
        try:
            self.github = Github(self.gh_token)
            self.repo = self.github.get_repo(self.repo_name)
        except GithubException as e:
            print(f"GitHub API connection failed: {str(e)}")
            raise

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

    def get_changed_files(self, pr):
        """Get all changed files with their contents"""
        changed_files = []
        base_sha = pr.base.sha
        head_sha = pr.head.sha
        
        comparison = self.repo.compare(base_sha, head_sha)
        
        for file in comparison.files:
            if file.status != 'modified':
                continue
                
            if not any(file.filename.endswith(ext) for ext in ['.py', '.js', '.java']):
                print(f"Skipping non-code file: {file.filename}")
                continue
                
            try:
                # Get file content at HEAD
                head_content = self.repo.get_contents(file.filename, ref=head_sha).decoded_content.decode()
                
                # Get file content at BASE
                base_content = self.repo.get_contents(file.filename, ref=base_sha).decoded_content.decode()
                
                changed_files.append({
                    'filename': file.filename,
                    'head_content': head_content,
                    'base_content': base_content,
                    'patch': file.patch
                })
                print(f"Found changed file: {file.filename}")
                
            except Exception as e:
                print(f"Couldn't get contents for {file.filename}: {str(e)}")
                continue
                
        return changed_files

    def analyze_code(self, file_content):
        """Analyze code using the model with better prompt engineering"""
        try:
            # Add context to help the model generate better suggestions
            prompt = f"""
            Analyze this code for potential improvements. Focus on:
            - Syntax errors
            - Code style violations
            - Performance optimizations
            - Security vulnerabilities
            - Best practices
            
            Code to review:
            {file_content}
            
            Suggestions:
            """
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True
            )
            suggestion = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Clean up the suggestion
            suggestion = suggestion.strip()
            suggestion = re.sub(r'<[^>]+>', '', suggestion)  # Remove any HTML-like tags
            suggestion = re.sub(r'^\W+', '', suggestion)  # Remove leading non-word chars
            
            return suggestion if suggestion and len(suggestion) > 10 else None
            
        except Exception as e:
            print(f"Failed to analyze code: {str(e)}")
            return None

    def post_comment(self, pr, filename, line_number, suggestion):
        """Post a single review comment with better validation"""
        try:
            # Validate the suggestion first
            if not suggestion or len(suggestion) < 10:
                print(f"Skipping invalid suggestion: {suggestion}")
                return False
                
            if any(tag in suggestion.lower() for tag in ['<e0>', '<msg>', '<code>']):
                print(f"Skipping suggestion with invalid tags: {suggestion}")
                return False
                
            print(f"Posting comment on {filename} line {line_number}: {suggestion}")
            
            pr.create_review_comment(
                body=f"🔍 **Code Review Suggestion**: {suggestion}",
                commit=pr.head,
                path=filename,
                line=line_number,
            )
            time.sleep(1)  # Rate limiting
            return True
        except GithubException as e:
            print(f"GitHub API error: {str(e)}")
            return False
        except Exception as e:
            print(f"Failed to post comment: {str(e)}")
            return False

    def run(self):
        """Main execution flow"""
        try:
            pr = self.get_pr_details()
            changed_files = self.get_changed_files(pr)
            
            if not changed_files:
                print("No changed code files found")
                return
            
            for file in changed_files:
                print(f"\nAnalyzing {file['filename']}...")
                
                # Analyze the entire file content first
                file_suggestion = self.analyze_code(file['head_content'])
                if file_suggestion:
                    print(f"General suggestion for file: {file_suggestion}")
                    if not self.post_comment(pr, file['filename'], 1, file_suggestion):
                        print("Failed to post general file suggestion")
                
                # Then analyze individual changed lines
                if file['patch']:
                    for line_num, line in self.parse_patch(file['patch']):
                        line_suggestion = self.analyze_code(line)
                        if line_suggestion:
                            print(f"Line {line_num} suggestion: {line_suggestion}")
                            if not self.post_comment(pr, file['filename'], line_num, line_suggestion):
                                print(f"Failed to post comment for line {line_num}")
                                continue
            
            print("\nReview completed successfully")
            
        except Exception as e:
            print(f"\nError in code review process: {str(e)}")
            raise

    def parse_patch(self, patch_text):
        """Parse patch to get changed lines and their numbers"""
        if not patch_text:
            return []
            
        lines = patch_text.split('\n')
        current_line = None
        results = []
        
        for line in lines:
            if line.startswith('@@ '):
                # Parse the line number from diff header
                parts = line.split(' ')
                if len(parts) >= 3:
                    line_info = parts[2].split(',')[0][1:]
                    try:
                        current_line = int(line_info)
                    except ValueError:
                        current_line = None
            elif current_line is not None:
                if line.startswith('+') and not line.startswith('++'):
                    results.append((current_line, line[1:]))
                    current_line += 1
                elif line.startswith(' '):
                    current_line += 1
                    
        return results

if __name__ == "__main__":
    try:
        print("Starting code review process...")
        reviewer = CodeReviewer()
        reviewer.run()
        print("Process completed")
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        exit(1)