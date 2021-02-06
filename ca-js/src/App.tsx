import React, { Component, RefObject } from "react";
import { MultiImgCAModel } from "./multi-img-ca";
import { createGrid } from "./multi-img-ca-grid";
import { createImgTexture } from "./img-texture";

export class App extends Component {
  // I don't know why, but we need to declare 'this.mount' for typescript to compile
  mount: any;
  targetImage: RefObject<HTMLImageElement>;

  constructor(props: Object) {
    super(props);
    // the target image we will grow with CA
    this.targetImage = React.createRef<HTMLImageElement>();
  }

  async componentDidMount() {
    //const model: MultiImgCAModel = await MultiImgCAModel.loadModel("model");
    if (!this.targetImage.current) {
      console.error("where's my image?");
    }
    document.body.appendChild((await createImgTexture(this.targetImage.current as HTMLImageElement, {frameHeight: 200, frameWidth: 200})).view);

  }

  render() {
    return <div>
      <div ref={(mount) => (this.mount = mount)} />
      <img src="images/emoji_u1f36d.png" ref={this.targetImage}/>
      </div>;
  }
}
