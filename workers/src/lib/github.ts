const OWNER = 'iamseungpil';
const REPO = 'claude-code-study-submissions';
const WORKFLOW = 'collect-submission.yml';

export async function triggerSubmissionCollection(
  pat: string,
  week: number,
  userId: string,
  githubUrl: string,
): Promise<void> {
  try {
    const resp = await fetch(
      `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/dispatches`,
      {
        method: 'POST',
        headers: {
          Accept: 'application/vnd.github+json',
          Authorization: `Bearer ${pat}`,
          'User-Agent': 'claude-code-study-worker',
        },
        body: JSON.stringify({
          ref: 'main',
          inputs: { week: String(week), user_id: userId, github_url: githubUrl },
        }),
      },
    );
    if (!resp.ok) {
      console.error(`GitHub dispatch failed (${resp.status})`);
    }
  } catch (err) {
    console.error('GitHub dispatch error:', err);
  }
}
