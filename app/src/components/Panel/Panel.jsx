import React from "react";
import "./panel.css";
import Drawer from "../Drawer/Drawer";

const Panel = () => {
  return (
    <div className="container-panel">
      <div className="left-panel">
        <Drawer />
      </div>
    </div>
  );
};

export default Panel;
