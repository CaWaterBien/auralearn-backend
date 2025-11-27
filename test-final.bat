@echo off
echo ================================
echo   FINAL AURABOT TEST
echo ================================

timeout /t 3 /nobreak > nul

echo 1. Testing server health...
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:8000/api/aurabot/health > temp_result.txt
set /p HEALTH_CODE=<temp_result.txt
if "%HEALTH_CODE%"=="200" (
    echo ✅ HEALTH ENDPOINT: SUCCESS
) else (
    echo ❌ HEALTH ENDPOINT: FAIL ^(Code: %HEALTH_CODE%^)
)

echo.
echo 2. Testing session status...
curl -s -o nul -w "%%{http_code}" "http://127.0.0.1:8000/api/aurabot/session-status?session_id=test123" > temp_result.txt
set /p SESSION_CODE=<temp_result.txt
if "%SESSION_CODE%"=="200" (
    echo ✅ SESSION ENDPOINT: SUCCESS
) else (
    echo ❌ SESSION ENDPOINT: FAIL ^(Code: %SESSION_CODE%^)
)

echo.
echo 3. Testing ask endpoint...
curl -s -X POST -H "Content-Type: application/json" -d "{\"session_id\":\"test123\",\"question\":\"What is HTML?\"}" -o nul -w "%%{http_code}" http://127.0.0.1:8000/api/aurabot/ask > temp_result.txt
set /p ASK_CODE=<temp_result.txt
if "%ASK_CODE%"=="200" (
    echo ✅ ASK ENDPOINT: SUCCESS
) else (
    echo ❌ ASK ENDPOINT: FAIL ^(Code: %ASK_CODE%^)
)

del temp_result.txt 2>nul

echo.
echo ================================
echo SYSTEM STATUS SUMMARY:
echo Health: %HEALTH_CODE%
echo Session: %SESSION_CODE%  
echo Ask: %ASK_CODE%
echo ================================
echo.
echo If all show ✅ SUCCESS, your AuraBot is working!
echo Test in activity page: http://localhost:3000/activity/1
pause
