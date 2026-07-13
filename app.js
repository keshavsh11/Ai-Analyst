require("dotenv").config();
const express = require("express");
const cors = require("cors");

const uploadRoute = require("./routes/uploadRoute");
const askRoute = require("./routes/askRoute");


const app = express();

app.use(cors({
    origin: "*"
}));
app.use(express.json());

///////////////////////////////////////////// Routes ///////////////////////////////////////
app.use("/upload", uploadRoute);
app.use("/ask", askRoute);

app.get("/", (req, res) => {
    res.json({
        message: "Node server running"
    });
});

const PORT = 5000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});