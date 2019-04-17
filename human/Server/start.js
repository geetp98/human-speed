var express = require('express'); 
var app = express(); 
const cors = require('cors');

app.use(cors());

app.listen(4000, function() { 
    console.log('server running on port 4000'); 
} )

app.get('/name', callName); 
  
function callName(req, res) { 

    var spawn = require("child_process").spawn; 

    var process = spawn('python',["./hello1.py", 
                            req.query.inputfileName,
                            req.query.yn,
                            req.query.yn2] );
    res.send("Successful");
} 