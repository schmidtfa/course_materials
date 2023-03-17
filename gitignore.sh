find . -type d -name "*data*" -print0 | xargs -0 echo "*/" > "$(dirname $0)/.gitignore"
