import os
import json
import time
import re
import torch
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
        
        # Files to skip (reviewer's own files)
        self.skip_files = [
            'src/automl_review.py',
            '.github/workflows/'
        ]
        
        # Initialize models with proper error handling
        try:
            print("Initializing CodeReviewer model...")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "microsoft/codereviewer",
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise
        
        # Initialize GitHub client with retry
        self.github = Github(
            self.gh_token,
            timeout=60,
            per_page=100,
            retry=3
        )
        self.repo = self.github.get_repo(self.repo_name)

    def should_skip_file(self, filename):
        """Check if file should be skipped"""
        return any(skip in filename for skip in self.skip_files)

    def get_pr_details(self):
        """Fetch PR details with proper error handling"""
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
        """Get changed files with robust error handling"""
        changed_files = []
        try:
            comparison = self.repo.compare(pr.base.sha, pr.head.sha)
            for file in comparison.files:
                if file.status not in ['modified', 'added']:
                    continue
                if self.should_skip_file(file.filename):
                    print(f"Skipping reviewer file: {file.filename}")
                    continue
                if not any(file.filename.endswith(ext) for ext in ['.py', '.js', '.java', '.ts', '.go']):
                    print(f"Skipping non-code file: {file.filename}")
                    continue
                
                try:
                    head_content = self.repo.get_contents(file.filename, ref=pr.head.sha).decoded_content.decode()
                    changed_files.append({
                        'filename': file.filename,
                        'head_content': head_content,
                        'patch': file.patch
                    })
                    print(f"Found changed file: {file.filename}")
                except Exception as e:
                    print(f"Couldn't get contents for {file.filename}: {str(e)}")
        except Exception as e:
            print(f"Failed to get changed files: {str(e)}")
            raise
        return changed_files

    def analyze_code(self, code_block):
        """Analyze code with proper model configuration"""
        try:
            prompt = f"""
            Analyze this code for IMPORTANT issues only (ignore formatting/whitespace):
            1. Actual bugs/errors
            2. Security vulnerabilities
            3. Performance issues
            4. Major style violations
            5. Architectural problems
            Ignore whitespace and formatting unless it affects functionality.
            
            Code:
            {code_block[:2000]}
            
            Critical Issues Found:
            """
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Strict quality filtering
            if (not suggestion or 
                len(suggestion.split()) < 8 or
                any(phrase in suggestion.lower() for phrase in [
                    'whitespace', 'blank line', 'indentation',
                    'space', 'formatting', 'extra line'
                ])):
                return None
                
            return suggestion
            
        except Exception as e:
            print(f"Error analyzing code: {str(e)}")
            return None

    def parse_patch(self, patch_text):
        """Parse patch to get meaningful changes"""
        if not patch_text:
            return []
            
        lines = patch_text.split('\n')
        current_line = None
        results = []
        
        for line in lines:
            if line.startswith('@@ '):
                parts = line.split(' ')
                if len(parts) >= 3:
                    try:
                        current_line = int(parts[2].split(',')[0][1:])
                    except ValueError:
                        current_line = None
            elif current_line is not None:
                if line.startswith('+') and not line.startswith('++'):
                    results.append((current_line, line[1:]))
                if line.startswith('+') or line.startswith(' '):
                    current_line += 1
                    
        return results

    def post_review(self, pr, suggestions):
        """Post review with all comments in one batch"""
        if not suggestions:
            print("No valid suggestions to post")
            return False
            
        try:
            comments = []
            for item in suggestions:
                comments.append({
                    'path': item['filename'],
                    'position': item['line_number'],
                    'body': f"🔍 **Code Review**:\n\n{item['suggestion']}\n\n**Impact**: {item['filename']} line {item['line_number']}"
                })
            
            pr.create_review(
                commit=pr.head.sha,
                body="Automated code review completed",
                event="COMMENT",
                comments=comments
            )
            print(f"Posted {len(comments)} comments successfully")
            return True
        except Exception as e:
            print(f"Failed to post review: {str(e)}")
            return False

    def run(self):
        """Main execution flow"""
        try:
            pr = self.get_pr_details()
            changed_files = self.get_changed_files(pr)
            
            if not changed_files:
                print("No changed code files found")
                return
            
            suggestions = []
            for file in changed_files:
                print(f"\nAnalyzing {file['filename']}...")
                
                if file['patch']:
                    for line_num, code in self.parse_patch(file['patch']):
                        suggestion = self.analyze_code(code)
                        if suggestion:
                            suggestions.append({
                                'filename': file['filename'],
                                'line_number': line_num,
                                'suggestion': suggestion
                            })
            
            if suggestions:
                self.post_review(pr, suggestions)
            else:
                print("\nNo significant issues found")
            
            print("\nReview completed successfully")
            
        except Exception as e:
            print(f"\nReview failed: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        print("Starting code review process...")
        reviewer = CodeReviewer()
        reviewer.run()
        print("Process completed")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        exit(1)