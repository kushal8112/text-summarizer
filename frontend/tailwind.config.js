module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          1: '#202123',
          2: '#343541',
          3: '#444654',
        },
        primary: {
          DEFAULT: '#19C37D',
          hover: '#1a7f64',
        }
      }
    },
  },
  plugins: [],
}
