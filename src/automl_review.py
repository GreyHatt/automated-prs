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
        
        # Files to skip (reviewer's own files)
        self.skip_files = [
            'src/automl_review.py',
            '.github/workflows/'
        ]
        
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

    def should_skip_file(self, filename):
        """Check if file should be skipped"""
        return any(skip in filename for skip in self.skip_files)

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
            if file.status != 'modified' and file.status != 'added':
                continue
                
            if self.should_skip_file(file.filename):
                print(f"Skipping reviewer file: {file.filename}")
                continue
                
            if not any(file.filename.endswith(ext) for ext in ['.py', '.js', '.java', '.ts', '.go']):
                print(f"Skipping non-code file: {file.filename}")
                continue
                
            try:
                # Get file content at HEAD
                head_content = self.repo.get_contents(file.filename, ref=head_sha).decoded_content.decode()
                
                changed_files.append({
                    'filename': file.filename,
                    'head_content': head_content,
                    'patch': file.patch
                })
                print(f"Found changed file: {file.filename}")
                
            except Exception as e:
                print(f"Couldn't get contents for {file.filename}: {str(e)}")
                continue
                
        return changed_files

    def analyze_code(self, file_content):
        """Analyze code using the model with focused prompt engineering"""
        try:
            # More specific prompt to focus on code quality
            prompt = f"""
            Analyze this code for specific improvements in these categories:
            1. BUGS - Actual code errors that will cause failures
            2. SECURITY - Potential security vulnerabilities
            3. PERFORMANCE - Optimizations for speed/memory
            4. STYLE - Code style violations (PEP8, etc)
            5. BEST PRACTICES - Better ways to implement
            
            Ignore whitespace and formatting unless it affects functionality.
            Provide concrete suggestions with explanations.
            
            Code:
            {file_content[:2000]}
            
            Analysis:
            """
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=5,
                early_stopping=True,
                temperature=0.5,  # Lower temperature for more focused results
                no_repeat_ngram_size=3
            )
            suggestion = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Filter suggestions
            suggestion = suggestion.strip()
            if not suggestion or len(suggestion.split()) < 5:
                return None
                
            # Remove generic suggestions
            bad_phrases = [
                "remove blank line",
                "add space",
                "extra whitespace",
                "formatting issue",
                "indentation"
            ]
            if any(phrase.lower() in suggestion.lower() for phrase in bad_phrases):
                return None
                
            return suggestion
            
        except Exception as e:
            print(f"Failed to analyze code: {str(e)}")
            return None

    def post_comments(self, pr, all_suggestions):
        """Post all comments in a single review with proper submission"""
        if not all_suggestions:
            print("No valid suggestions to post")
            return False

        try:
            # Create a review with all comments at once
            review_comments = []
            for item in all_suggestions:
                formatted_suggestion = f"""🚨 **Code Review - Important Suggestion**:
                
    {item['suggestion']}

    **Impact**: This is a significant issue that could affect functionality, security, or performance.
    """
                review_comments.append({
                    'path': item['filename'],
                    'position': item['line_number'],
                    'body': formatted_suggestion
                })

            # Submit the review with all comments
            pr.create_review(
                commit=pr.head,
                body="Automated code review with suggested changes",
                event="COMMENT",  # Use COMMENT instead of REQUEST_CHANGES to be less intrusive
                comments=review_comments
            )
            print(f"Successfully posted {len(review_comments)} comments")
            return True
        except GithubException as e:
            print(f"GitHub API error: {str(e)}")
            return False
        except Exception as e:
            print(f"Failed to post comments: {str(e)}")
            return False

    def parse_patch(self, patch_text):
        """Parse patch to get meaningful code changes with positions"""
        if not patch_text:
            return []
            
        lines = patch_text.split('\n')
        current_position = None
        results = []
        current_hunk_lines = []
        hunk_start_line = None
        
        for line in lines:
            if line.startswith('@@ '):
                # Save previous hunk if we have one
                if current_hunk_lines and hunk_start_line is not None:
                    code_block = '\n'.join([l[1:] if l.startswith('+') else l for l in current_hunk_lines])
                    if len(code_block.strip()) > 20:
                        results.append((hunk_start_line, code_block))
                
                # Start new hunk
                current_hunk_lines = []
                parts = line.split(' ')
                if len(parts) >= 3:
                    new_part = parts[2]
                    new_start = new_part.split(',')[0][1:]
                    try:
                        hunk_start_line = int(new_start)
                        current_position = hunk_start_line
                    except ValueError:
                        hunk_start_line = None
                        current_position = None
            
            elif current_position is not None:
                if line.startswith('+') or line.startswith(' '):
                    current_hunk_lines.append(line)
                    if line.startswith('+') and not line.startswith('++'):
                        current_position += 1
                elif line.startswith('-'):
                    current_hunk_lines.append(line)
        
        # Add the last hunk if it exists
        if current_hunk_lines and hunk_start_line is not None:
            code_block = '\n'.join([l[1:] if l.startswith('+') else l for l in current_hunk_lines])
            if len(code_block.strip()) > 20:
                results.append((hunk_start_line, code_block))
                    
        return results

    def run(self):
        """Main execution flow"""
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
                
                # Analyze the entire file for architectural issues
                file_suggestion = self.analyze_code(file['head_content'])
                if file_suggestion:
                    print(f"Found file-level issue: {file_suggestion[:100]}...")
                    all_suggestions.append({
                        'filename': file['filename'],
                        'line_number': 1,
                        'suggestion': file_suggestion
                    })
            
            # Post all suggestions in a single review
            if all_suggestions:
                print(f"\nPosting {len(all_suggestions)} significant suggestions...")
                if not self.post_comments(pr, all_suggestions):
                    print("Failed to post some comments")
            else:
                print("\nNo significant issues found - code looks good!")
            
            print("\nReview completed successfully")
            
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