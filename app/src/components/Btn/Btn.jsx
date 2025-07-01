import React from "react";
import "./btn.css";

const Btn = ({ type, onClick }) => {
  return (
    <div className="container-btn">
      {type === "generate" ? (
        <p className="btn-verify" onClick={onClick}>
          Vérifier
        </p>
      ) : (
        <p className="btn-retry" onClick={onClick}>
          Effacer
        </p>
      )}
    </div>
  );
};

export default Btn;
