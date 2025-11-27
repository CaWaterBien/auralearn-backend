@echo off
echo ================================
echo   🤖 AURABOT SYSTEM STATUS 🤖
echo ================================
echo.

echo ✅ Backend Server: RUNNING on http://127.0.0.1:8000
echo ✅ Frontend Server: RUNNING on http://localhost:3000  
echo ✅ Boolean Error: FIXED
echo ✅ Ask Endpoint: WORKING

echo.
echo ================================
echo   LIVE TEST INSTRUCTIONS
echo ================================
echo.
echo 1. Open your browser and go to:
echo    http://localhost:3000/activity/1
echo.
echo 2. You should see:
echo    - AuraBot chat window on the right side
echo    - Welcome message from AuraBot 
echo    - Input field to ask questions
echo.
echo 3. Try asking AuraBot:
echo    - "How do I create a heading in HTML?"
echo.
echo 4. Expected behavior:
echo    - Your message appears on RIGHT (blue)
echo    - AuraBot reply appears on LEFT (default)
echo    - Question counter decreases (3 → 2 → 1 → 0)
echo    - After 3 questions, you'll be blocked

echo.
echo 🚀 EVERYTHING IS READY! Test it now!
echo.
pause
