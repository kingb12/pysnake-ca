import React, { Component, RefObject } from "react";
import { MultiImgCAModel } from "./multi-img-ca";
import { createGrid } from "./multi-img-ca-grid";

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
    const model: MultiImgCAModel = await MultiImgCAModel.loadModel("model");
    if (!this.targetImage.current) {
      console.error("where's my image?");
    }
    document.body.appendChild((await createGrid(model, this.targetImage.current as HTMLImageElement, {frameHeight: 200, frameWidth: 200})).view);
  }

  render() {
    return <div>
      <div ref={(mount) => (this.mount = mount)} />
      <img src="https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u1f36d.png" ref={this.targetImage}/>
      </div>;
  }
}
