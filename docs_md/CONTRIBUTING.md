# Contributing to Trading Bot

## Branch Protection Policy

This repository uses branch protection to maintain code quality and prevent direct pushes to the main branch.

## Development Workflow

### 1. Create a New Branch

Before making any changes, create a new branch:

```bash
# Update your local main branch
git checkout master
git pull origin master

# Create and checkout a new feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features (e.g., `feature/add-new-indicator`)
- `fix/` - Bug fixes (e.g., `fix/correct-sharpe-calculation`)
- `docs/` - Documentation updates (e.g., `docs/update-readme`)
- `refactor/` - Code refactoring (e.g., `refactor/optimize-backtest`)

### 2. Make Your Changes

Edit files, test your changes, and commit:

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add adaptive regime detection to strategy"
```

### 3. Push Your Branch

```bash
# Push your branch to GitHub
git push origin feature/your-feature-name
```

### 4. Create a Pull Request

1. Go to: https://github.com/Levietduc1104/trading_bot
2. Click "Pull requests" → "New pull request"
3. Select your branch to merge into `master`
4. Fill in the PR description:
   - What changes were made
   - Why the changes were needed
   - How to test the changes
5. Click "Create pull request"

### 5. Code Review

- Wait for review and approval
- Make requested changes if needed
- Once approved, the PR can be merged

### 6. After Merge

```bash
# Switch back to main branch
git checkout master

# Pull the latest changes
git pull origin master

# Delete your local feature branch (optional)
git branch -d feature/your-feature-name
```

## Quick Reference

```bash
# Create new branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin feature/my-feature

# After PR is merged, clean up
git checkout master
git pull origin master
git branch -d feature/my-feature
```

## Protected Branch Rules

The `master` branch is protected with the following rules:

- ❌ No direct pushes to master
- ✅ Pull requests required
- ✅ At least 1 approval required (if enabled)
- ✅ All tests must pass (if configured)

## Running Tests Before Push

Always run the backtest execution before creating a PR:

```bash
# Run the complete system
python src/core/execution.py

# Verify outputs
ls -la output/data/trading_results.db
ls -la output/plots/trading_analysis.html
ls -la output/reports/
```

## Questions?

If you have questions about the contribution workflow, open an issue at:
https://github.com/Levietduc1104/trading_bot/issues
