import React, { Component } from "react";
import { MDBJumbotron, MDBBtn, MDBContainer, MDBRow, MDBCol, MDBIcon, MDBCardBody, MDBCardText,  MDBCardTitle } from "mdbreact";
import { CASimulation, CASimulationProps } from "./ca-simulation";
import { SimpleGridCA } from "./ca-grid";
import { ImageSelector } from "./image-selector";

export class MyArticle extends Component {
  constructor(props: any) {
    super(props)
  }

  render() {
    const simProps: CASimulationProps = {
      gridCA: new SimpleGridCA(),
      viewOpts: {width: 500, height: 500, maxFPS: 10},
      id: 'MyCoolCASimulation'
    }
    const MyCA = () => {
      return new CASimulation(simProps);
    }
    return (
      <MDBContainer className="my-1 py-0 text-center"> 
        <ArticleTitle title="Growing Image Informed Neural Cellular Automata" subtitle="Demos and Early Experiments"></ArticleTitle>
        {/* TODO: extract to side-by-side component? */}
        <MDBRow>
          <MDBCol xl="8">
            <CASimulation {...simProps}/> 
          </MDBCol>
          <MDBCol size="4">
            <ImageSelector {...{images: [
              "images/emoji_u1f0cf_40px.png",
              "images/emoji_u1f1f8_40px.png",
              "images/emoji_u1f1fd_40px.png",
              "images/emoji_u1f3c1_40px.png",
              "images/emoji_u1f3c6_40px.png",
              "images/emoji_u1f3d2_40px.png",
              "images/emoji_u1f3ec_40px.png",
              "images/emoji_u1f3f3_40px.png",
              "images/emoji_u1f3fa_40px.png",
              "images/emoji_u1f17e_40px.png",
              "images/emoji_u1f36b_40px.png",
            ]}}>
            </ImageSelector>
          </MDBCol>
        </MDBRow>
        
      </MDBContainer>
    )
  }
}

/**
 * A title banner for an article
 * @param props 
 */
export const ArticleTitle: React.FC<{title: string, subtitle: string}> = (props: {title: string, subtitle: string}) => {
  return (
  <MDBCardBody>
        <MDBCardTitle className="h1 my-0 py-0">
          {props.title}
        </MDBCardTitle>
        <p className="blue-text my-1 font-weight-bold">
          {props.subtitle}
        </p>
  </MDBCardBody>
  );
}

export default MyArticle;