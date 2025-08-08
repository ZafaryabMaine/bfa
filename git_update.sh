#!/bin/bash

# Ensure we are in a Git repository
if [ ! -d ".git" ]; then
    echo "Error: Not a Git repository."
    exit 1
fi

# Fetch the latest changes from the remote
git fetch
fetch_status=$?
if [ $fetch_status -ne 0 ]; then
    echo "Error: 'git fetch' failed."
    exit $fetch_status
fi

# Check if the local branch is behind the remote
local_branch=$(git symbolic-ref --short HEAD)
upstream="${local_branch}@{upstream}"
ahead=$(git rev-list --count $upstream..$local_branch)
behind=$(git rev-list --count $local_branch..$upstream)

if [ $behind -ne 0 ]; then
    echo "Local branch is behind the remote branch. Pulling changes..."
    git pull --rebase
    pull_status=$?
    if [ $pull_status -ne 0 ]; then
        echo "Error: 'git pull' failed."
        exit $pull_status
    fi
elif [ $ahead -ne 0 ]; then
    echo "Local branch is ahead of the remote branch."
else
    echo "Local branch is up to date with the remote branch."
fi

# Add all changes except those in .gitignore
git add -A --verbose
add_status=$?
if [ $add_status -ne 0 ]; then
    echo "Error: 'git add' failed."
    exit $add_status
fi

# Commit with a dynamic message or a default one if not provided
commit_message=${1:-"Quick update"}
git commit -m "$commit_message"
commit_status=$?
if [ $commit_status -ne 0]; then
    echo "Error: 'git commit' failed."
    exit $commit_status
fi

# Push to the current branch's remote
git push --verbose
push_status=$?
if [ $push_status -ne 0]; then
    echo "Error: 'git push' failed."
    exit $push_status
fi

echo "Changes pushed successfully."