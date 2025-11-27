#!/bin/bash

echo "Setting up AuraLearn database..."
echo "================================"

# Run migrations
echo "Running database migrations..."
php artisan migrate

# Seed the database
echo "Seeding database with sample data..."
php artisan db:seed

# Clear cache
echo "Clearing application cache..."
php artisan cache:clear
php artisan config:clear
php artisan route:clear

echo "================================"
echo "Database setup complete!"
echo ""
echo "Default admin credentials:"
echo "Email: admin@auralearn.local"
echo "Password: Admin123!"
echo ""
echo "Sample courses and code examples have been created."
