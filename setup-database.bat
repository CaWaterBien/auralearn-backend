@echo off
echo Setting up AuraLearn database...
echo ================================

echo Running database migrations...
php artisan migrate

echo.
echo Seeding database with sample data...
php artisan db:seed

echo.
echo Clearing application cache...
php artisan cache:clear
php artisan config:clear
php artisan route:clear

echo.
echo ================================
echo Database setup complete!
echo.
echo Default admin credentials:
echo Email: admin@auralearn.local
echo Password: Admin123!
echo.
echo Sample courses and code examples have been created.
pause
