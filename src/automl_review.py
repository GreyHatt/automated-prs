import os
import json
import time
import re
from github import Github, GithubException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
import torch

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
        
        # Initialize models with proper configuration
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
        
        # Initialize GitHub client with retry logic
        try:
            self.github = Github(
                self.gh_token,
                timeout=60,
                per_page=100,
                retry=3
            )
            self.repo = self.github.get_repo(self.repo_name)
        except GithubException as e:
            print(f"GitHub API connection failed: {str(e)}")
            raise

    def analyze_code(self, file_content):
        """Analyze code using the model with proper configuration"""
        try:
            prompt = f"""
            Analyze this code for specific improvements in these categories:
            1. BUGS - Actual code errors that will cause failures.
            2. SECURITY - Potential security vulnerabilities
            3. PERFORMANCE - Optimizations for speed/memory
            4. STYLE - Code style violations (PEP8, etc)
            5. BEST PRACTICES - Better ways to implement
            
            Ignore whitespace and formatting unless it affects functionality.
            Provide concrete suggestions with explanations.
            
            Code:
            {file_content[:2000]}
            
            Significant Issues Found:
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
                    do_sample=False,  # Disable sampling for deterministic results
                    temperature=1.0,   # Neutral temperature when do_sample=False
                    no_repeat_ngram_size=3
                )
            
            suggestion = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Strict filtering of suggestions
            suggestion = suggestion.strip()
            if not suggestion:
                return None
                
            # Skip if suggestion is too short or doesn't contain actionable items
            if len(suggestion.split()) < 8:
                return None
                
            # Skip formatting-related suggestions
            formatting_phrases = [
                'whitespace', 'blank line', 'indentation', 
                'space', 'formatting', 'extra line', 'remove line'
            ]
            if any(phrase in suggestion.lower() for phrase in formatting_phrases):
                return None
                
            return suggestion
            
        except Exception as e:
            print(f"Failed to analyze code: {str(e)}")
            return None

    def post_comments(self, pr, all_suggestions):
        """Post all comments with proper error handling"""
        if not all_suggestions:
            print("No valid suggestions to post")
            return False

        try:
            # Prepare comments with proper positions
            comments = []
            for item in all_suggestions:
                formatted_suggestion = f"""🚨 **Code Review**:
                
{item['suggestion']}

**Impact**: This could affect {item['filename']}
"""
                comments.append({
                    'path': item['filename'],
                    'position': item['line_number'],
                    'body': formatted_suggestion
                })

            # Create review in one API call
            pr.create_review(
                commit=pr.head.sha,
                body="Automated code review suggestions",
                event="COMMENT",
                comments=comments
            )
            print(f"Posted {len(comments)} comments successfully")
            return True
            
        except GithubException as e:
            print(f"GitHub API error: {str(e)}")
            # Fallback to individual comments if batch fails
            return self._post_comments_individually(pr, all_suggestions)
        except Exception as e:
            print(f"Failed to post comments: {str(e)}")
            return False

    def _post_comments_individually(self, pr, all_suggestions):
        """Fallback method for posting comments one by one"""
        success_count = 0
        for item in all_suggestions:
            try:
                pr.create_review_comment(
                    body=f"🔍 **Code Review**: {item['suggestion']}",
                    commit=pr.head,
                    path=item['filename'],
                    line=item['line_number'],
                )
                success_count += 1
                time.sleep(2)  # Rate limiting
            except Exception as e:
                print(f"Failed to post comment for {item['filename']} line {item['line_number']}: {str(e)}")
        
        print(f"Posted {success_count}/{len(all_suggestions)} comments individually")
        return success_count > 0

    def run(self):
        """Main execution flow with error handling"""
        try:
            pr = self.get_pr_details()
            changed_files = self.get_changed_files(pr)
            
            if not changed_files:
                print("No changed code files found")
                return
            
            # Collect all suggestions first
            all_suggestions = []
            for file in changed_files:
                print(f"\nAnalyzing {file['filename']}...")
                
                # Analyze individual changed hunks
                if file['patch']:
                    for line_num, code_block in self.parse_patch(file['patch']):
                        suggestion = self.analyze_code(code_block)
                        if suggestion:
                            print(f"Found issue at line {line_num}: {suggestion[:100]}...")
                            all_suggestions.append({
                                'filename': file['filename'],
                                'line_number': line_num,
                                'suggestion': suggestion
                            })
            
            # Post all suggestions
            if all_suggestions:
                self.post_comments(pr, all_suggestions)
            else:
                print("\nNo significant issues found")
            
            print("\nReview completed")
            
        except Exception as e:
            print(f"\nCritical error: {str(e)}")
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