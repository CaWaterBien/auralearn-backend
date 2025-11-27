@echo off
echo Testing AI Validation System
echo ==============================
echo.

echo Step 1: Testing with simple text
curl -s -X POST "http://127.0.0.1:8000/api/activities/2/submit" -H "Content-Type: application/json" -d "{\"user_code\":\"Hello World\",\"time_spent_minutes\":1}"

echo.
echo.
echo Step 2: Checking if server is responding
curl -s -X GET "http://127.0.0.1:8000/api/activities/2/status"

echo.
echo.
echo ==============================
pause
