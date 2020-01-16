:: current path
%~d0
cd %~dp0
:: input commit message
set /p commit_msg=Please input commit message:
:: git status
git status
:: git pull
git pull
:: add all changing
git add -A
:: commit
git commit -am "%commit_msg%"
:: git push
git push
:: make a pause
pause