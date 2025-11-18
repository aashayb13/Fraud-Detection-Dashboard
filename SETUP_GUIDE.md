# VS Code Setup Guide

This guide will help you set up your VS Code environment to work with this repository.

## Prerequisites

- **Git**: [Download Git](https://git-scm.com/downloads)
- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **VS Code**: [Download VS Code](https://code.visualstudio.com/)

## Initial Setup (First Time Only)

### Step 1: Clone the Repository

Open a terminal/command prompt on your desktop and run:

```bash
# Navigate to where you want the project
cd ~/Desktop  # or C:\Users\YourName\Desktop on Windows

# Clone the repository
git clone https://github.com/Texasdada13/transaction-monitoring.git

# Navigate into the project
cd transaction-monitoring
```

### Step 2: Open in VS Code

```bash
# Open the project in VS Code
code .
```

Or manually: Open VS Code → File → Open Folder → Select the `transaction-monitoring` folder

### Step 3: Install Recommended Extensions

When you open the project, VS Code should prompt you to install recommended extensions. Click **Install All**.

If not prompted, manually install:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- GitLens (eamodio.gitlens)

### Step 4: Run the Setup Script

**On Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup.ps1
```

This script will:
- Create a virtual environment (`venv/`)
- Install all Python dependencies
- Configure Git settings

### Step 5: Select Python Interpreter

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type "Python: Select Interpreter"
3. Choose the one that shows `./venv/bin/python` or `.\venv\Scripts\python.exe`

## Daily Workflow

### Opening the Project

1. Open VS Code
2. File → Open Recent → Select `transaction-monitoring`
3. Open a new terminal in VS Code (`Ctrl+~` or `Cmd+~`)
4. The virtual environment should automatically activate (you'll see `(venv)` in the terminal)

### Syncing with GitHub

VS Code is configured to auto-fetch changes every 3 minutes. To manually sync:

**Pull latest changes:**
```bash
git pull origin main
```

**Check current branch:**
```bash
git branch
```

**Switch branches:**
```bash
git checkout branch-name
```

### If Virtual Environment Doesn't Activate Automatically

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```powershell
.\venv\Scripts\Activate.ps1
```

## Troubleshooting

### Virtual Environment Not Activating

1. Check that the `.vscode/settings.json` file exists
2. Restart VS Code
3. Manually select the Python interpreter (Step 5 above)
4. Open a new terminal

### Git Not Syncing

1. Check your internet connection
2. Verify Git credentials:
   ```bash
   git config --list
   ```
3. Try manually pulling:
   ```bash
   git pull origin main
   ```

### Python Packages Missing

Activate the virtual environment and reinstall:
```bash
# Activate venv first (see above)
pip install -r requirements.txt
```

### Permission Errors (Windows)

If you get execution policy errors with PowerShell, run PowerShell as Administrator:
```powershell
Set-ExecutionPolicy RemoteSigned
```

## Project Structure

```
transaction-monitoring/
├── venv/                  # Virtual environment (auto-created)
├── .vscode/               # VS Code settings (auto-configured)
├── app/                   # Main application code
├── config/                # Configuration files
├── dashboard/             # Dashboard components
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
├── run.py                 # Main entry point
└── setup.sh / setup.ps1   # Setup scripts
```

## Useful VS Code Shortcuts

- `Ctrl+~` / `Cmd+~`: Toggle terminal
- `Ctrl+Shift+P` / `Cmd+Shift+P`: Command palette
- `Ctrl+P` / `Cmd+P`: Quick file open
- `F5`: Run/Debug
- `Ctrl+Shift+G` / `Cmd+Shift+G`: Git panel

## Git Branch Information

You're currently working on branch: `claude/troubleshoot-vscode-setup-011CUXzB7kJ5LY52av11hoFj`

To see all branches:
```bash
git branch -a
```

To create a new branch:
```bash
git checkout -b your-branch-name
```

## Need Help?

- Check the [ARCHITECTURE.md](./ARCHITECTURE.md) file for system documentation
- Review Python virtual environment docs: https://docs.python.org/3/tutorial/venv.html
- VS Code Python docs: https://code.visualstudio.com/docs/python/python-tutorial
