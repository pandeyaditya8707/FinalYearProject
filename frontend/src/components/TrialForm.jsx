import { useState } from "react";
import { IoIosArrowBack } from "react-icons/io";
import StepOne from "./CountryDropdown";
import StepTwo from "./FileUpload";

function TrialForm({ onBack }) {
  const [step, setStep] = useState(1);
  const [country, setCountry] = useState("");

  const handleNext = () => setStep((prev) => prev + 1);
  const handleBack = () => (step === 1 ? onBack() : setStep((prev) => prev - 1));

  return (
    <div className="relative w-[400px] p-6 rounded-lg border border-[#221b42] shadow-lg">
      {/* Back Button */}
      <button
        onClick={handleBack}
        className="absolute top-3 left-4 flex items-center text-gray-300 hover:text-white text-sm"
      >
        <IoIosArrowBack className="mr-1" size={16} /> {/* Left Arrow Icon */}
        {step === 1 ? "Back to Home" : "Back"}
      </button>

      {/* Step Tracker */}
      <div className="flex items-center justify-center mb-6 mt-6"> {/* Add margin-top for spacing */}
        <div
          className={`w-8 h-8 flex items-center justify-center rounded-full text-white font-bold ${
            step === 1 ? "bg-primary" : "bg-transparent border-2 border-gray-500"
          }`}
        >
          1
        </div>
        <div className="w-12 h-1 bg-gray-500 mx-2"></div>
        <div
          className={`w-8 h-8 flex items-center justify-center rounded-full text-white font-bold ${
            step === 2 ? "bg-primary" : "bg-transparent border-2 border-gray-500"
          }`}
        >
          2
        </div>
      </div>

      {/* Step Components */}
      {step === 1 ? <StepOne country={country} setCountry={setCountry} onNext={handleNext} /> : <StepTwo country={country}/>}
    </div>
  );
}

export default TrialForm;
