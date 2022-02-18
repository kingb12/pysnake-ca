import React, { Component, RefObject, useEffect } from "react";
import { SimpleGridCA } from "./ca-grid";
import { CASimulation, CASimulationProps } from "./ca-simulation";
import { MultiImgCAModel } from "./multi-img-ca";
// @ts-ignore
import * as dt from "distill-template";
import { DtArticle, MyP } from "./dt-components";

import JumbotronPage from "./Article";


export const App: React.FC = () => {
  return (
    // TODO: a header with links that take you back to my home page
    <JumbotronPage/>
  );
};



export class App2 extends Component {
  // I don't know why, but we need to declare 'this.mount' for typescript to compile
  mount: any;
  targetImage: RefObject<HTMLImageElement>;
  distill: HTMLElement;

  constructor(props: Object) {
    super(props);
    // the target image we will grow with CA
    this.targetImage = React.createRef<HTMLImageElement>();
    const script = document.createElement('script');
    script.src = "https://distilll.pub/template.v2.js";
    script.async = true;
    this.distill = script;
  }

  async componentDidMount() {
    const model: MultiImgCAModel = await MultiImgCAModel.loadModel("model");
    console.log(dt);
    document.body.appendChild(this.distill);
    if (!this.targetImage.current) {
      //console.error("where's my image?");
    }
  }

  async componentWillUnmount() {
      document.body.removeChild(this.distill);
  }

  render() {
    
    const props: CASimulationProps = {
      gridCA: new SimpleGridCA(),
      viewOpts: {width: 400, height: 400, maxFPS: 10},
      id: 'MyCoolCASimulation'
    }
    return <div>
      <DtArticle>
        <MyP>sgfjnsdlkfjsdnflkasjdnflksjdnflakjsdnflkj</MyP>
      </DtArticle>
      </div>
  }
}