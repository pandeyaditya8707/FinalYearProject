import { useState } from "react";
import "./App.css";
import Navbar from "./components/Navbar";
import TrialForm from "./components/TrialForm";

function App() {
  const [showForm, setShowForm] = useState(false);

  return (
    <div className="h-screen bg-[#000000]" >
      <Navbar />

      

      <div className="container h-full mx-auto flex items-center justify-center">
        <div className="flex flex-col items-center text-center">
          {/* Gradient Text */}
          <h1
            className="text-6xl p-4 font-bold text-transparent bg-clip-text"
            style={{
              backgroundImage:
                "linear-gradient(89.95deg, #feb484 3.13%, #a686f2 66.86%, #592ac8 123.2%)",
            }}
          >
            PlateVision
          </h1>

          {/* Subheading */}
          <p className="text-lg text-gray-300 my-2">
            Experience unparalleled accuracy with our AI-powered solution for
            precise vehicle number plate detection.
          </p>
          <div className="flex py-4 mt-2">
            {!showForm ? (
              <button
                onClick={() => setShowForm(true)}
                className="bg-white text-lg text-black py-2 px-6 rounded-md font-medium hover:bg-black hover:text-white transition duration-300 border-2 border-white"
              >
                Get free trial
              </button>
            ) : (
              <TrialForm onBack={() => setShowForm(false)} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
