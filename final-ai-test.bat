@echo off
echo 🤖 AI Validation System - FINAL TEST
echo =====================================
echo.

echo Testing the submission endpoint properly...
echo.

curl -s -X POST "http://127.0.0.1:8000/api/activities/2/submit" -H "Content-Type: application/json" -d "{\"user_code\":\"<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Hello World</h1></body></html>\",\"time_spent_minutes\":2}"

echo.
echo.
echo Testing simple code:
echo.

curl -s -X POST "http://127.0.0.1:8000/api/activities/2/submit" -H "Content-Type: application/json" -d "{\"user_code\":\"<h1>Hello</h1>\",\"time_spent_minutes\":1}"

echo.
echo.
echo =====================================
echo Test Complete!
echo =====================================
pause
