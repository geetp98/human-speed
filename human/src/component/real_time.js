import React, { Component } from 'react';
import axios from 'axios';
import '../main.scss';
import '../style/styles.css';
import 'bootstrap/dist/css/bootstrap.css';

class RealTime extends Component {

  constructor(){
    super()
    this.state={
      inputFileName:'0',
      y_n:'n',
      y_n2:'n',
      checkedout:false,
      checkedgraph:false,
      real_time:false,
      isLoading: false,
      isSubmitted: false
    }
  }

  handleRealTime = () => {
    console.log('Clicked')
      this.setState({isSubmitted: true})
      const URL = `http://localhost:4000/name?inputfileName=${this.state.inputFileName}&yn=${this.state.y_n}&yn2=${this.state.y_n2}`
      axios.get(URL)
      .then( res => {
        setTimeout(() => {
          this.setState({isLoading: true});
          console.log(res);
        } , 10000);
        this.setState({isLoading:false});
      })
      .catch(err => {
        console.log("Error")
      })
  }

  handleCheckout = () => {
    if(this.state.checkedout)
    {
      this.setState({y_n:'n'})
    }
    else
    {
      this.setState({y_n:'y'})
    }
    this.setState({checkedout: !this.state.checkedout})
  }

  handleCheckgraph = () => {
    if(this.state.checkedgraph)
    {
      this.setState({y_n2:'n'})
    }
    else
    {
      this.setState({y_n2:'y'})
    }
    this.setState({checkedgraph: !this.state.checkedgraph})
  }

  render() {
    return (
      <div>
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

      <div style={{alignItems:'center', fontSize:'1.5rem', padding: '1.5rem', marginTop:'9rem'}}>
        <div style={{color: 'white', textAlign:'center'}}> Would you like to suppress streaming of the output?: &nbsp;
          <input type="checkbox" style={{height: '1.5rem', width:'1.5rem'}} onChange={this.handleCheckout}></input>
        </div>
        </div>
        <div style={{alignItems:'center', fontSize:'1.5rem', padding: '1.5rem'}}>
        <div style={{color: 'white',textAlign:'center'}}> Would you like to see graphs generated from the input? &nbsp;
          <input type="checkbox"  style={{height: '1.5rem', width:'1.5rem'}} onChange={this.handleCheckgraph}></input>
        </div>
        </div>
        <div style={{padding:'5rem 30rem'}}>
        <button type="button" className="btn btn-primary btn-lg btn-block" onClick={this.handleRealTime} style={{borderRadius:'30px'}}>Start</button>
        </div>
      </div>

    );
  }
}

export default RealTime;
