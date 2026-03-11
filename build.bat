@echo off
:: CRITICAL: The FBX SDK requires Python 3.10 (e.g., the 'mocap_studio' environment).
:: Do NOT use the 'omnicontrol' environment (Python 3.7) or the FBX import will fail in the exe.
call C:\Users\bxdpo\Miniconda3\Scripts\activate.bat mocap_studio
pyinstaller --clean MoCapStudio_v2.spec
