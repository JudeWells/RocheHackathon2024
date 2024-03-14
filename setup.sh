# Run this file if you want to set up the environment for the first time
# on a local machine (rather than on saturn cloud)

#!/bin/bash

# Check if git LFS is installed
if ! command -v git-lfs &> /dev/null
then
    echo "Git LFS is not installed. Installing..."

    # Detect the operating system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install git-lfs
    elif [[ "$OSTYPE" == "msys" ]]; then
        # Windows (Git Bash)
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
        apt-get install git-lfs
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        sudo apt-get install git-lfs
    else
        echo "Unsupported operating system. Please install Git LFS manually."
        echo "https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage"
        exit 1
    fi

    echo "Git LFS installed successfully."
fi

echo "current directory:"
pwd
echo "directory contents:"
ls
echo "installing virtual environment"
python3 -m venv venv_prot
source venv_prot/bin/activate
echo "current python path after activating virtual environment:"
which python
echo "directory from which install command pip install -r requirements.txt will be run:"
pwd
pip install -r requirements.txt
echo "requirements install finished"
git lfs install
git lfs fetch
git lfs pull
echo "lfs pull operation completed, updated directory contents:"
ls
echo "output of git lfs track:"
git lfs track
echo "You can now test if the install was sucessful by running the following command:"
echo "python src/train.py"