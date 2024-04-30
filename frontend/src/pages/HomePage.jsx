import Lottie from "lottie-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import sample from "../assets/4800099-hd_1920_1080_30fps.mp4";
import animationData from "../assets/Animation - 1713599754822.json";
import ChatBox from "../components/ChatBox";
import Footer from "../components/Footer";
import Navbar from "../components/Navbar";

const HomePage = () => {
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const toggleChatbox = () => {
    setIsOpen(!isOpen);
  };
  const navigateToDashboard = () => {
    navigate("dashboard");
  };
  return (
    <div>
      <Navbar />
      <button
        class="chat-button fixed bottom-10 right-10 z-10"
        onClick={toggleChatbox}
      >
        {!isOpen && <Lottie animationData={animationData} className="w-52" />}
      </button>
      <button
        class="chat-button fixed bottom-10 right-20 z-10"
        onClick={toggleChatbox}
      >
        {isOpen && (
          <Lottie animationData={animationData} className="mr-[350px] w-52" />
        )}
      </button>
      {isOpen && <ChatBox />}
      <div className="hero min-h-screen">
        <video
          className="videoTag"
          autoPlay
          loop
          muted
          style={{
            width: "100%",
            float: "left",
            top: 0,
            padding: "none",
            position: "fixed",
            zIndex: -1,
          }}
        >
          <source src={sample} type="video/mp4" />
        </video>
        <div className="hero-content text-center">
          <div className="max-w-md">
            <h1 className="mb-5 text-5xl font-bold">Hello there</h1>
            <p className="mb-5">
              Provident cupiditate voluptatem et in. Quaerat fugiat ut assumenda
              excepturi exercitationem quasi. In deleniti eaque aut repudiandae
              et a id nisi.
            </p>
            <button className="btn btn-primary" onClick={navigateToDashboard}>
              Get Started
            </button>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default HomePage;
