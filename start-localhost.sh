#!/usr/bin/env bash
echo "==================================================="
echo "Starting AI Resume Analyzer on http://localhost:3000"
echo "==================================================="
if [ -d "node_modules" ]; then
  npm run dev
else
  npm install && npm run dev || node standalone-localhost-server.cjs
fi
