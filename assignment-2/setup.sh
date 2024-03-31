# to enable this script : chmod +x setup.sh
# usage : source setup.sh

PY_ENV_NAME=".venv"

if [ -d $PY_ENV_NAME ]; then
    echo "Python environment exists."
else
    echo "Python environment does not exist. Creating virtual environment."
    python3 -m venv $PY_ENV_NAME
    echo "Python virtual environment created."
fi

echo "Activating python virtual environment."
source $PY_ENV_NAME/bin/activate
pip install -r requirements.txt
echo "Python virtual environment activated."
echo /$PY_ENV_NAME > .gitignore