#!/bin/bash
# Script to clean up repository and push to GitHub

set -e  # Exit on error

echo "Creating a clean directory..."
mkdir -p ~/lead-measure-clean

echo "Copying files (excluding .env, .git, and unnecessary files)..."
rsync -av --exclude='.env' --exclude='.git' --exclude='__pycache__' --exclude='.DS_Store' --exclude='*.pyc' ./ ~/lead-measure-clean/

echo "Copying .gitignore to the clean directory..."
cp .gitignore ~/lead-measure-clean/

echo "Initializing new Git repository..."
cd ~/lead-measure-clean
git init

echo "Adding files to Git..."
git add .

echo "Committing files..."
git commit -m "Initial commit with clean repository"

# Uncomment and update the following lines when ready to push
echo "Setting up remote repository..."
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git): " REPO_URL
git remote add origin $REPO_URL

echo "Pushing to GitHub..."
git push -u origin main || git push -u origin master

echo "Done! Your code has been pushed to GitHub without sensitive data in the history."
echo "You can now continue working in: ~/lead-measure-clean" 