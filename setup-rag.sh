#!/bin/bash

echo "================================"
echo "   AURALEARN RAG SETUP SCRIPT"
echo "================================"
echo

echo "[1/6] Installing PHP packages..."
composer install

echo
echo "[2/6] Installing Node.js packages..."
npm install

echo
echo "[3/6] Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env file created from example"
fi

echo
echo "[4/6] Running database migrations..."
php artisan migrate

echo
echo "[5/6] Creating sample RAG data..."
php artisan rag:create-samples

echo
echo "[6/6] Ingesting course content..."
php artisan rag:ingest-courses

echo
echo "================================"
echo "   RAG SETUP COMPLETE!"
echo "================================"
echo
echo "IMPORTANT: Make sure to set these environment variables in your .env file:"
echo
echo "Required for OpenAI embeddings (fallback):"
echo "OPENAI_API_KEY=your_openai_api_key_here"
echo
echo "Nebius configuration is already set up!"
echo
echo "To test the system:"
echo "1. Start the Laravel server: php artisan serve"
echo "2. Start the frontend: cd ../capstone-app && npm run dev"
echo "3. Test AuraBot health: php artisan tinker then \"app(\App\Services\NebiusClient::class)->testConnection()\""
echo
echo "Default admin credentials:"
echo "Email: admin@auralearn.local"
echo "Password: Admin123!"
echo

