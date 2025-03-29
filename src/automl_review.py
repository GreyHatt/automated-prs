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
        
        # Files to skip
        self.skip_files = [
            'src/automl_review.py',
            '.github/workflows/'
        ]
        
        # Initialize models
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
        
        # Initialize GitHub client
        self.github = Github(
            self.gh_token,
            timeout=60,
            per_page=100,
            retry=3
        )
        self.repo = self.github.get_repo(self.repo_name)

    def get_pr_details(self):
        """Get PR details from GitHub event payload"""
        try:
            with open(self.event_path, 'r') as f:
                event_data = json.load(f)
            
            # Extract PR number from the event data
            pr_number = event_data['number']
            pr = self.repo.get_pull(pr_number)
            
            return pr
            
        except Exception as e:
            print(f"Error retrieving PR details: {str(e)}")
            raise

    def analyze_code(self, code_block):
        """Enhanced code analysis to catch syntax errors and logical issues"""
        try:
            # More specific prompt to catch errors
            prompt = f"""
            Analyze this Python code for:
            1. Syntax errors (missing operators, incomplete statements)
            2. Logical errors (incorrect operations)
            3. Undefined variables
            4. Missing return values
            5. Function definition issues

            Ignore whitespace and formatting unless it affects functionality.
            
            Code:
            {code_block}
            
            Issues Found:
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
            
            # Filter for meaningful suggestions
            if not suggestion or len(suggestion.split()) < 5:
                return None
                
            # Skip generic suggestions
            skip_phrases = [
                'add documentation',
                'improve variable names',
                'add comments',
                'formatting',
                'whitespace'
            ]
            if any(phrase.lower() in suggestion.lower() for phrase in skip_phrases):
                return None
                
            return suggestion
            
        except Exception as e:
            print(f"Error analyzing code: {str(e)}")
            return None

    def parse_patch(self, patch_text):
        """Parse patch to get code changes with context"""
        if not patch_text:
            return []
            
        lines = patch_text.split('\n')
        current_line = None
        results = []
        current_hunk = []
        
        for line in lines:
            if line.startswith('@@ '):
                if current_hunk and current_line is not None:
                    code_block = '\n'.join([l[1:] if l.startswith('+') else l for l in current_hunk])
                    if len(code_block.strip()) > 0:
                        results.append((current_line - len(current_hunk) + 1, code_block))
                current_hunk = []
                parts = line.split(' ')
                if len(parts) >= 3:
                    try:
                        current_line = int(parts[2].split(',')[0][1:])
                    except ValueError:
                        current_line = None
            elif current_line is not None:
                if line.startswith('+') or line.startswith(' '):
                    current_hunk.append(line)
                    if line.startswith('+') and not line.startswith('++'):
                        current_line += 1
        
        # Add the last hunk
        if current_hunk and current_line is not None:
            code_block = '\n'.join([l[1:] if l.startswith('+') else l for l in current_hunk])
            if len(code_block.strip()) > 0:
                results.append((current_line - len(current_hunk) + 1, code_block))
                
        return results

    def run(self):
        """Main execution flow with enhanced error detection"""
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
                        # Focus on lines with actual code changes
                        if any(op in code for op in ['+', '-', '*', '/', '=', 'return']):
                            suggestion = self.analyze_code(code)
                            if suggestion:
                                print(f"Found issue at line {line_num}: {suggestion[:100]}...")
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