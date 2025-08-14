const path = require("path");
const webpack = require("webpack");

module.exports = {
  devtool: "cheap-eval-source-map",
  entry: "./src/index.js",
  output: {
    path: path.join(__dirname, "dist"),
    filename: "bundle.js",
    // publicPath: "/"
  },
  resolve: {
    alias: {
      atomize: path.join(__dirname, "src")
    }
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/
      },
      { test: /\.css$/, use: "css-loader"},
      { test: /\.ts$/,  use: 'ts-loader' }
    ],
  },
  plugins: [
    new webpack.DefinePlugin({
      "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV)
    })
  ],
  devServer: {
    contentBase: "docs/",
    historyApiFallback: true
  }
};
