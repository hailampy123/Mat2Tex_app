import React from "react";
import {BrowserRouter, Routes, Route, Navigate} from "react-router-dom";

import Home from "./pages/HomePage";
import About from "./pages/AboutPage";
import Docs from "./pages/DocsPage";

function App() {
  return (
    <BrowserRouter>
    <Routes>
      <Route path="/home" element={<Home/>} />
      <Route
        path="*"
        element={<Navigate replace to="/home" />}
      />
      <Route path="/about" element={<About />} />
      <Route path="/docs" element={<Docs />} />
    </Routes>
    </BrowserRouter>
  );
}

export default App;
