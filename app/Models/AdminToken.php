<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class AdminToken extends Model
{
    protected $table = 'admin_tokens';

    protected $fillable = [
        'admin_id', 'token', 'last_used_at', 'expires_at'
    ];

    public $timestamps = true;
} 