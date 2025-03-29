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

    def post_comment(self, pr, filename, line_number, suggestion):
        """Post a single review comment with proper commit reference"""
        try:
            # Get the commit list for the PR
            commits = pr.get_commits()
            if commits.totalCount == 0:
                print("No commits found in PR")
                return False
                
            # Use the most recent commit
            commit = commits[commits.totalCount - 1]
            
            print(f"Attempting to post comment on {filename} line {line_number}")
            
            # Create the comment
            pr.create_review_comment(
                body=f"🔍 **Code Review**: {suggestion}",
                commit=commit,
                path=filename,
                line=line_number,
            )
            time.sleep(2)  # More conservative rate limiting
            return True
        except GithubException as e:
            print(f"GitHub API error: {str(e)}")
            return False
        except Exception as e:
            print(f"Failed to post comment: {str(e)}")
            return False

    def parse_patch(self, patch_text):
        """Parse patch to get meaningful code changes"""
        if not patch_text:
            return []
            
        lines = patch_text.split('\n')
        current_line = None
        results = []
        min_context_lines = 3  # Include surrounding context
        
        for line in lines:
            if line.startswith('@@ '):
                parts = line.split(' ')
                if len(parts) >= 3:
                    new_part = parts[2]
                    new_start = new_part.split(',')[0][1:]
                    try:
                        current_line = int(new_start)
                    except ValueError:
                        current_line = None
            elif current_line is not None:
                if line.startswith('+') and not line.startswith('++'):
                    # Capture some context around changes
                    context = []
                    idx = lines.index(line)
                    for i in range(max(0, idx-min_context_lines), min(idx+min_context_lines+1, len(lines))):
                        context_line = lines[i]
                        if not context_line.startswith('@') and not context_line.startswith('++') and not context_line.startswith('--'):
                            context.append(context_line[1:] if context_line.startswith('+') else context_line)
                    
                    code_block = '\n'.join(context)
                    if len(code_block.strip()) > 10:  # Only include meaningful changes
                        results.append((current_line, code_block))
                    current_line += 1
                elif line.startswith(' '):
                    current_line += 1
                    
        return results

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
                
                # Analyze individual changed lines first
                if file['patch']:
                    line_suggestions = []
                    for line_num, line in self.parse_patch(file['patch']):
                        if len(line.strip()) == 0:
                            continue
                            
                        suggestion = self.analyze_code(line)
                        if suggestion:
                            print(f"Line {line_num} suggestion: {suggestion}")
                            line_suggestions.append((line_num, suggestion))
                    
                    # Post line comments in reverse order (avoids line number shifting issues)
                    for line_num, suggestion in reversed(line_suggestions):
                        if not self.post_comment(pr, file['filename'], line_num, suggestion):
                            print(f"Failed to post comment for line {line_num}")
                            continue
                
                # Then analyze the entire file for general suggestions
                file_suggestion = self.analyze_code(file['head_content'])
                if file_suggestion:
                    print(f"General file suggestion: {file_suggestion}")
                    if not self.post_comment(pr, file['filename'], 1, file_suggestion):
                        print("Failed to post general file suggestion")
            
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