import os
import json
from github import Github
from google.cloud import aiplatform
from dotenv import load_dotenv
from diff_match_patch import diff_match_patch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

class CodeReviewer:
    def __init__(self):
        self.gh_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY")
        self.event_path = os.getenv("GITHUB_EVENT_PATH")
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.region = os.getenv("GCP_REGION")
        self.endpoint_id = os.getenv("ENDPOINT_ID")
        
        # Initialize HuggingFace model and tokenizer for codereviewer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.region)
        self.endpoint = aiplatform.Endpoint(self.endpoint_id)
        
        # Initialize GitHub client
        self.github = Github(self.gh_token)
        self.repo = self.github.get_repo(self.repo_name)
        
        # Initialize diff tools
        self.dmp = diff_match_patch()

    def get_pr_details(self):
        """Fetch the PR details from GitHub"""
        with open(self.event_path, 'r') as f:
            event_data = json.load(f)
        pr_number = event_data['number']
        return self.repo.get_pull(pr_number)

    def parse_diff(self, diff_text):
        """Parse unified diff to extract changed lines with positions"""
        changes = []
        lines = diff_text.split('\n')
        file_path = None
        line_number = None
        chunk_start = None
        
        for line in lines:
            if line.startswith('+++ b/'):
                file_path = line[6:]
            elif line.startswith('@@ '):
                parts = line.split(' ')
                new_part = parts[2]
                new_start, new_count = map(int, new_part[1:].split(','))
                chunk_start = new_start
                current_line = new_start
            elif line.startswith('+') and not line.startswith('++'):
                changes.append({
                    'file_path': file_path,
                    'line_number': current_line,
                    'content': line[1:]
                })
                current_line += 1
            elif line.startswith('-') and not line.startswith('--'):
                continue
            elif line.startswith(' '):
                current_line += 1
        
        return changes

    def analyze_code(self, code_snippets):
        """Send code snippets to HuggingFace model for analysis (local model or GCP)"""
        results = []
        for snippet in code_snippets:
            inputs = self.tokenizer(snippet["content"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            outputs = self.model.generate(**inputs)
            suggestions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'file_path': snippet['file_path'],
                'line_number': snippet['line_number'],
                'content': snippet['content'],
                'suggestions': suggestions
            })
        
        return results

    def post_comments(self, pr, analysis_results):
        """Post review comments to GitHub PR"""
        for result in analysis_results:
            pr.create_review_comment(
                body="\n".join([f"🔍 **Suggestion**: {s}" for s in result['suggestions']]),
                commit_id=pr.head.sha,
                path=result['file_path'],
                line=result['line_number'],
            )

    def run(self):
        pr = self.get_pr_details()
        files = pr.get_files()
        
        all_changes = []
        for file in files:
            if not file.filename.endswith(('.py', '.js', '.java', '.go', '.ts')):  # Add more extensions as needed
                continue
            
            if file.patch:
                changes = self.parse_diff(file.patch)
                all_changes.extend(changes)
        
        if all_changes:
            analysis_results = self.analyze_code(all_changes)
            if analysis_results:
                self.post_comments(pr, analysis_results)
                print(f"Posted {len(analysis_results)} review comments")
            else:
                print("No suggestions from the model")
        else:
            print("No code changes found to analyze")

if __name__ == "__main__":
    reviewer = CodeReviewer()
    reviewer.run()