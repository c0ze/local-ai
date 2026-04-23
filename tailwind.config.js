/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts}',
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ['ui-serif', 'Georgia', 'Cambria', '"Times New Roman"', 'serif'],
        sans: ['ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'sans-serif'],
      },
      colors: {
        cream: {
          50: '#faf9f5',
          100: '#f5f1ea',
          200: '#ede8dc',
          300: '#ddd6c5',
          400: '#b8b1a0',
        },
        ink: {
          400: '#8a857d',
          500: '#6b6760',
          600: '#4a4742',
          700: '#2f2d29',
          900: '#191917',
        },
        accent: {
          DEFAULT: '#c15f3c',
          hover: '#a64d2e',
          subtle: '#e8d5c8',
        },
      },
    },
  },
  plugins: [],
};
