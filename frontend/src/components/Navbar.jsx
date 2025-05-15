import React from 'react';

const Navbar = () => {
  return (
    <nav className="fixed top-10 left-1/2 transform -translate-x-1/2 z-10 w-[90%] max-w-6xl">
      <div className="backdrop-blur-md bg-black/50 shadow-md px-4 py-3 rounded-xl">
        <div className="flex justify-center items-center">
          <h1 className="text-center font-bold text-sm sm:text-lg md:text-xl">
            <span className="text-orange-400">Cogniflow AI:</span>{" "}
            <span className="text-purple-400">Advanced Adaptive Traffic Signal System and Vechile classification</span>
          </h1>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;