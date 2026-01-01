# GitHub Branch Protection Setup Guide

Follow these steps to enable branch protection on your repository.

## Step 1: Navigate to Repository Settings

1. Go to your repository: https://github.com/Levietduc1104/trading_bot
2. Click on **Settings** tab (top right)
3. In the left sidebar, click **Branches** (under "Code and automation")

## Step 2: Add Branch Protection Rule

1. Click **Add rule** or **Add branch protection rule**
2. In "Branch name pattern", enter: `master`

## Step 3: Configure Protection Rules

### ✅ Required Settings (Recommended):

**Protect matching branches:**
- ☑️ **Require a pull request before merging**
  - ☑️ Require approvals: 1 (optional if you're the only developer)
  - ☐ Dismiss stale pull request approvals when new commits are pushed
  - ☐ Require review from Code Owners (optional)

- ☑️ **Require status checks to pass before merging** (if you have CI/CD)
  - ☐ Require branches to be up to date before merging

- ☑️ **Require conversation resolution before merging** (recommended)

- ☐ Require signed commits (optional, more secure)

- ☐ Require linear history (optional)

**Rules applied to everyone including administrators:**
- ☑️ **Include administrators** (highly recommended - even you can't bypass)

**Additional settings:**
- ☐ Allow force pushes (keep unchecked)
- ☐ Allow deletions (keep unchecked)

## Step 4: Save Changes

1. Scroll to the bottom
2. Click **Create** or **Save changes**

## Step 5: Test the Protection

Try to push directly to master:

```bash
# This should FAIL
git checkout master
git push origin master
# Error: protected branch hook declined
```

## Alternative: Using GitHub CLI

If you have GitHub CLI installed (`gh`):

```bash
# Install gh if needed
brew install gh  # macOS
# or download from https://cli.github.com/

# Login
gh auth login

# Create branch protection rule
gh api repos/Levietduc1104/trading_bot/branches/master/protection \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  --field required_pull_request_reviews='{"required_approving_review_count":1}' \
  --field enforce_admins=true \
  --field restrictions=null
```

## Verification

After setup, you should see:
- A shield icon next to "master" branch
- "Protected" label on the branch
- Pull request requirement when trying to merge

## Working with Protected Branch

From now on, follow this workflow:

```bash
# 1. Create feature branch
git checkout -b feature/my-changes

# 2. Make changes and commit
git add .
git commit -m "Description"

# 3. Push to GitHub
git push origin feature/my-changes

# 4. Create Pull Request on GitHub
# Visit: https://github.com/Levietduc1104/trading_bot/pulls

# 5. Review and merge through GitHub UI

# 6. Update local master
git checkout master
git pull origin master
```

## Troubleshooting

### "Can't push to master"
✅ This is correct\! Create a feature branch instead.

### "Need approval but I'm the only developer"
You can either:
- Disable "Require approvals" in settings
- Self-approve your own PRs (if not enforcing admin rules)

### "Want to bypass protection for urgent fix"
⚠️ Not recommended\! But if necessary:
1. Temporarily disable protection in settings
2. Push the fix
3. Re-enable protection immediately

## Resources

- [GitHub Docs: Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub Docs: Pull Requests](https://docs.github.com/en/pull-requests)

