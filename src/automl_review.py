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
            timeout=120,  # Increased timeout to 120 seconds
            per_page=100,
            retry=3
        )
        self.repo = self.github.get_repo(self.repo_name)
    
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
    
    def analyze_code(self, code_block):
        """Enhanced code analysis with better error detection"""
        try:
            # More structured prompt with examples
            prompt = f"""
            Analyze this code for potential issues. Provide specific recommendations in this format:
            
            [ISSUE TYPE]: [DESCRIPTION]
            [SEVERITY]: [HIGH/MEDIUM/LOW]
            [LOCATION]: [LINE NUMBER OR FUNCTION NAME]
            [SUGGESTION]: [CONCRETE FIX]
            
            Check for:
            1. Syntax errors (missing brackets, semicolons, invalid operators)
            2. Function issues (wrong parameters, missing returns, incorrect usage)
            3. Performance problems (nested loops, unnecessary computations)
            4. Logical errors (incorrect conditions, wrong variable usage)
            5. Security vulnerabilities (SQL injection, hardcoded secrets)
            6. Code smells (long methods, duplicate code)
            
            Example findings:
            - SYNTAX ERROR: Missing closing bracket in if statement
            - FUNCTION ISSUE: Function 'calculate' expects 3 parameters but called with 2
            - OPTIMIZATION: Loop can be simplified using map()
            - LOGIC ERROR: Condition will always evaluate to true
            
            Code to analyze:
            {code_block}
            
            Findings:
            """
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048  # Increased for more context
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,  # Increased for more detailed analysis
                    num_beams=5,
                    early_stopping=True,
                    temperature=0.7,  # Added for more creative suggestions
                    no_repeat_ngram_size=3
                )
            
            suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # More sophisticated filtering
            if not suggestion:
                return None
                
            # Skip only truly generic suggestions
            skip_phrases = [
                'no issues found',
                'looks good',
                'no problems detected',
                'add documentation',
                'improve variable names',
                'add comments',
                'formatting',
                'whitespace'
            ]
            
            # Keep suggestions that contain specific issue indicators
            keep_indicators = [
                'error', 'issue', 'warning', 'problem', 
                'optimiz', 'improve', 'suggest', 'consider',
                'vulnerability', 'risk', 'bug'
            ]
            
            if any(phrase.lower() in suggestion.lower() for phrase in skip_phrases):
                return None
                
            if not any(indicator.lower() in suggestion.lower() for indicator in keep_indicators):
                return None
                
            return suggestion
            
        except Exception as e:
            print(f"Error analyzing code: {str(e)}")
            return None
    
    def validate_code_syntax(self, code_block, language):
        """Basic syntax validation before sending to model"""
        try:
            if language == 'python':
                # Simple Python syntax check
                try:
                    ast.parse(code_block)
                except SyntaxError as e:
                    return f"Syntax Error: {str(e)}"
                    
            elif language in ['javascript', 'typescript']:
                # Basic JS syntax check patterns
                if re.search(r'function\s+\w+\s*\([^)]*\)\s*{[^}]*$', code_block):
                    return "Syntax Error: Missing closing brace in function"
                if re.search(r'\([^)]*$', code_block):
                    return "Syntax Error: Unclosed parentheses"
                    
            return None
        except Exception:
            return None

    def parse_patch(self, patch_text):
        """Enhanced patch parsing with better context capture"""
        if not patch_text:
            return []
            
        lines = patch_text.split('\n')
        current_line = None
        results = []
        current_hunk = []
        context_lines = 3  # Number of surrounding lines to include
        
        for i, line in enumerate(lines):
            if line.startswith('@@ '):
                if current_hunk and current_line is not None:
                    # Include surrounding context
                    start_idx = max(0, i - context_lines)
                    end_idx = min(len(lines), i + context_lines + 1)
                    context = lines[start_idx:end_idx]
                    
                    code_block = '\n'.join([
                        l[1:] if l.startswith('+') else l 
                        for l in current_hunk + context
                        if not l.startswith('@')
                    ])
                    
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
        
        # Add the last hunk with context
        if current_hunk and current_line is not None:
            code_block = '\n'.join([l[1:] if l.startswith('+') else l for l in current_hunk])
            if len(code_block.strip()) > 0:
                results.append((current_line - len(current_hunk) + 1, code_block))
                
        return results

    def run(self):
        """Enhanced execution flow with better error detection"""
        try:
            pr = self.get_pr_details()
            changed_files = self.get_changed_files(pr)
            
            if not changed_files:
                print("No changed code files found")
                return
            
            suggestions = []
            for file in changed_files:
                print(f"\nAnalyzing {file['filename']}...")
                file_ext = os.path.splitext(file['filename'])[1][1:].lower()
                
                if file['patch']:
                    for line_num, code in self.parse_patch(file['patch']):
                        # First do basic syntax validation
                        syntax_error = self.validate_code_syntax(code, file_ext)
                        if syntax_error:
                            suggestions.append({
                                'filename': file['filename'],
                                'line_number': line_num,
                                'suggestion': syntax_error,
                                'type': 'syntax'
                            })
                            continue
                            
                        # Then do deeper analysis
                        suggestion = self.analyze_code(code)
                        if suggestion:
                            print(f"Found issue at line {line_num}: {suggestion[:100]}...")
                            suggestions.append({
                                'filename': file['filename'],
                                'line_number': line_num,
                                'suggestion': suggestion,
                                'type': 'analysis'
                            })
            
            if suggestions:
                self.post_review(pr, suggestions)
            else:
                print("\nNo significant issues found")
            
            print("\nReview completed successfully")
            
        except Exception as e:
            print(f"\nReview failed: {str(e)}")
            raise

    def post_review(self, pr, suggestions):
        """Post suggestions as review comments on the pull request"""
        try:
            for suggestion in suggestions:
                # Skip trivial suggestions (e.g., blank line issues)
                if "blank line" in suggestion['suggestion'].lower() or "extra space" in suggestion['suggestion'].lower():
                    continue

                # Debugging: Log the suggestion before posting
                print(f"Posting review for: {suggestion}")

                # Check that the line_number and path are correct
                print(f"Posting comment at line {suggestion['line_number']} in {suggestion['filename']}")

                pr.create_review_comment(
                    body=f"Issue: {suggestion['suggestion']}",
                    commit_id=pr.head.sha,  # Ensure this is correct
                    path=suggestion['filename'],
                    position=suggestion['line_number']
                )
            print("Successfully posted review comments.")
        except GithubException as e:
            print(f"Failed to post comments: {str(e)}")
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