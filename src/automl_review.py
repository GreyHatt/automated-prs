import os
from google.cloud import automl_v1beta1 as automl
from github import Github

def get_diff_content():
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(os.getenv("GITHUB_REPOSITORY"))
    pr = repo.get_pull(int(os.getenv("GITHUB_EVENT_PATH").split('/')[-1]))
    diff = pr.get_files()
    diff_content = ''
    for file in diff:
        diff_content += file.raw_data
    return diff_content

def analyze_code(content):
    client = automl.PredictionServiceClient()
    model_path = f"projects/{os.getenv("GCP_PROJECT_ID")}/locations/{os.getenv("REGION")}/models/{os.getenv("MODEL_ID")}"
    payload = {"text_snippet": {"content": content, "mime_type": "text/plain"}}
    response = client.predict(name=model_path, payload=payload)
    return response

def post_review_comments(analysis_results):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(os.getenv("GITHUB_REPOSITORY"))
    pr = repo.get_pull(int(os.getenv("GITHUB_EVENT_PATH").split('/')[-1]))
    for result in analysis_results:
        pr.create_review_comment(
            body=result["suggestion"],
            commit_id=pr.head.sha,
            path=result["file"],
            position=result["position"]
        )

def main():
    diff_files = get_diff_content()
    analysis_results = []
    for file in diff_files:
        if file.filename.endswith('.py'):
            content = file.patch
            analysis = analyze_code(content)
            for prediction in analysis.payload:
                analysis_results.append({
                    "file": file.filename,
                    "position": prediction.position,
                    "suggestion": prediction.suggestion,
                })
    post_review_comments(analysis_results)

if __name__ == "__main__":
    main()