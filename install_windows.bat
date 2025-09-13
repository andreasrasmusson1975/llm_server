@echo off
setlocal

set REPO_URL=https://git@github.com/andreasrasmusson1975/llm_server.git
git clone %REPO_URL%
cd llm_server
echo Creating environment...
call python -m venv env
call env\Scripts\activate.bat

echo Upgrading pip in venv...
env\Scripts\python.exe -m pip install --upgrade pip

echo Installing llm_server (no deps) into venv...
env\Scripts\python.exe -m pip install . --no-deps

echo Installing core dependencies into venv...
env\Scripts\python.exe -m pip install "numpy<2"
env\Scripts\python.exe -m pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
env\Scripts\python.exe -m pip install transformers==4.56.1 accelerate==1.0.1 bitsandbytes==0.43.2
env\Scripts\python.exe -m pip install fastapi==0.111.0 uvicorn==0.30.1 azure-identity==1.15.0 azure-keyvault-secrets==4.8.0 pyyaml==6.0

echo Installation complete.
pause
endlocal
