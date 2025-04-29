export const environment = {
  production: true,
  apiUrl: process.env['FLASK_APP_API_URL'] || 'https://nagsabot.onrender.com' // Use environment variable or fallback
}; 