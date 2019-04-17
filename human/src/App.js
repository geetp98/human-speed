import React, { Component } from 'react';
import './App.css';
import './main.scss';
import UploadVideo from './component/upload_video';
import RealTime from './component/real_time';
import icon from './ouri.ico';

class App extends Component {

  constructor()
  {
    super()
    this.state = {
      isClicked: false,
      isRealTime:false
    }
  }

  handleRealTime = () =>{
    this.setState({isClicked:true});
    this.setState({isRealTime:true});
  }

  handleRecordedVideo = () =>{
    this.setState({isClicked:true});
    this.setState({isRealTime:false});
  }

  handleHome = () => {
    this.setState({isClicked:false})
  }

  render() {
    return (
      <div>
        {/* <nav className="navbar navbar-inverse" style={{backgroundColor: '#0069D9'}}>
          <div className="container-fluid">
            <div className="navbar-header" onClick={this.handleHome} style={{marginRight:'500px', marginLeft:'-100px', color:'white', fontSize:'1.5rem'}}>
              Human Speed Detection
            </div>
            <ul className="nav navbar-nav">
              <li className="active"><button className="btn btn-light" onClick={this.handleHome} style={{borderRadius:'30px'}}>Home</button></li>
            </ul>
            <ul className="nav navbar-nav navbar-right">
              <li><button className="btn btn-light" style={{borderRadius:'30px', marginRight:'-350px'}} data-toggle="modal" data-target="#myModal2">About US</button></li>
            </ul>
            <ul className="nav navbar-nav navbar-right">
              <li><button className="btn btn-light use" style={{borderRadius:'30px', height:'100%'}} data-toggle="modal" data-target="#myModal">How to use?</button></li>
            </ul>
          </div>
        </nav> */}

        <nav className="navbar navbar-expand-md navbar-dark bg-primary">
              <div className="navbar-collapse collapse w-100 order-1 order-md-0 dual-collapse2">
                  <div className="navbar-nav mr-auto">
                      <div className="nav-item active" style={{marginBottom:'-0.3rem', marginTop:'-0.3rem'}}>
                      {/* <img src={icon} style={{height:'2rem', width:'2rem', marginTop:'.7rem'}} alt="HSD icon"></img>  */}
                      <button className="navbar-brand mx-auto btn" onClick={this.handleHome} style={{marginLeft:'40%'}}> <span className='navbar-icon'><img src={icon} style={{height:'2.5rem', width:'2.5rem', marginBottom:'0.5rem'}} alt="HSD icon"></img></span> Human Speed Detection</button>

                      </div>
                  </div>
              </div>

              <div className="navbar-collapse collapse w-100 order-3 dual-collapse2">
                  <ul className="navbar-nav ml-auto">
                      <li className="nav-item">
                          <button className="nav-link btn use" style={{ height:'100%', fontWeight:'normal', color:'white'}} onClick={this.handleHome}>Home</button>
                      </li>
                      <li className="nav-item">
                          <button className="nav-link btn use" style={{ height:'100%', fontWeight:'normal', color:'white'}} data-toggle="modal" data-target="#myModal2">About</button>
                      </li>
                      <li className="nav-item">
                        <button className="btn use" style={{ height:'100%', fontWeight:'normal', color:'white'}} data-toggle="modal" data-target="#myModal">How to use?</button>
                      </li>
                  </ul>
              </div>
          </nav>
        {(!this.state.isClicked)?(<div>
          <h3 style={{textAlign:'center', color:'white', fontSize:'4vw'}}> Human Speed Detection</h3>
          <div className="modal w3-container w3-center w3-animate-top" id="myModal">
            <div className="modal-dialog modal-lg">
              <div className="modal-content">

                {/* <!-- Modal Header --> */}
                <div className="modal-header" style={{backgroundColor:"#283593", display:'flex'}}>
                  <h4 className="modal-title" style={{flex:'4', color:'white'}}>How To Use</h4>
                  <button type="button" className="close" data-dismiss="modal" style={{color: 'white'}}>&times;</button>
                </div>

                {/* <!-- Modal body --> */}
                <div className="modal-body" style={{fontSize: '1.2rem', textAlign:'left'}}>

                Open the website and then choose your desired option. <br/>There are two options displayed on the screen.
                <br/>1. Real-time : The camera of your laptop will start capturing the video.
                <br/>2. Upload-Video : The browser will guide you to upload your desired file.
                <br/>
                <br/>After choosing the appropriate option, you will be guided to another video.
                <br/>In case of Real-time
                <br/>You will have to choose from the following two options
                <br/>1. Would you like to suppress streaming of the output? (y/n) - This means that the output of the code will be saved, but it won’t be shown to user. So enter ‘y’ or ‘n’ according to your choice.
                <br/>2. Would you like to see graphs generated from the input? (y/n) - This asks the user whether he wants to see the graphs generated from the files. So enter ‘y’ or ‘n’ according to your choice.

                <br/>In case of Upload-Video
                <br/>Apart from the above mentioned options, you will be asked to browse and choose the appropriate files you want to upload.

                </div>

                {/* <!-- Modal footer --> */}
                <div className="modal-footer">
                  <button type="button" className="btn btn-danger" data-dismiss="modal">Close</button>
                </div>

              </div>
            </div>
          </div>

          <div className="modal w3-container w3-center w3-animate-top" id="myModal2">
            <div className="modal-dialog modal-lg">
              <div className="modal-content">

                {/* <!-- Modal Header --> */}
                <div className="modal-header" style={{backgroundColor:"#283593", display:'flex'}}>
                  <h4 className="modal-title" style={{flex:'4', color:'white'}}>About</h4>
                  <button type="button" className="close" data-dismiss="modal" style={{color: 'white'}}>&times;</button>
                </div>

                {/* <!-- Modal body --> */}
                <div className="modal-body" style={{fontSize: '1.2rem', textAlign:'left'}}>

                <p> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The following project is title “Human Speed Detection”.
                 It has been done as a part of the course “Software Engineering”,
                 under the guidance of Professor Saurabh Tiwari.
                 Goal is to measure speed of a moving human without using any external device other than a camera and a processing unit, where processing unit can be a PC (or a mobile device in an extended idea).
                 Objectives are making a code as simple and explainable as possible, given that it is a project in the domain of machine learning, and still producing maximum output.</p>

                </div>

                {/* <!-- Modal footer --> */}
                <div className="modal-footer">
                  <button type="button" className="btn btn-danger" data-dismiss="modal">Close</button>
                </div>

              </div>
            </div>
          </div>


          <div className="center">
          <button type="button" className="btn btn-primary btn-lg btn-block button" onClick={this.handleRealTime} style={{borderRadius:'30px'}}>Real Time</button>

          <button type="button" className="btn btn-primary btn-lg btn-block button" onClick={this.handleRecordedVideo} style={{borderRadius:'30px'}}>Upload Video</button>
          </div>
        </div>):
        (
          (this.state.isRealTime)?<RealTime handleHome={ this.handleHome } />:<UploadVideo handleHome={ this.handleHome }/>
        )}
    </div>
    );
  }
}

export default App;
