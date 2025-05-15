import React from "react";

function StepOne({ country, setCountry, onNext }) {
    return (
      <>
        <label htmlFor="country" className="block text-white text-lg mb-2">
          Select Country:
        </label>
        <select
          id="country"
          value={country}
          onChange={(e) => setCountry(e.target.value)}
          className="w-full p-2 rounded-md bg-gray-900 bg-opacity-85 text-white focus:outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="" disabled>
            Choose your country
          </option>
          <option value="US">United States</option>
          <option value="IN">India</option>
          <option value="UK">United Kingdom</option>
          <option value="CA">Canada</option>
        </select>
        <button
          onClick={onNext}
          disabled={!country}
          className="mt-4 w-full bg-primary text-white py-2 rounded-md hover:bg-[#592ac8] transition duration-300 disabled:opacity-50"
        >
          Proceed
        </button>
      </>
    );
  }
  
  export default StepOne;
  