import React, { Component } from "react";
import { MDBBtn, MDBBtnGroup, MDBContainer, MDBIcon, MDBLink, MDBMask, MDBView } from "mdbreact";
import { propTypes } from "react-bootstrap/esm/Image";


interface ImageSelectorState {
    selectedSrc: string
}
interface ClickableImageState {
    selected?: boolean
}
export class ClickableImage extends Component {
    onSelect: (src: string) => void;
    src: string;
    state: ClickableImageState

    constructor(props: {src: string, selected?: boolean, onSelect: (src: string) => void}) {
        super(props);
        this.select = this.select.bind(this);
        this.onSelect = props.onSelect;
        this.src = props.src;
        this.state = {selected: props.selected || false};
        this.setState({selected: props.selected});
    }

    async select() {
        // this.setState({selected: !!!this.state.selected});
        this.setState({selected: true}, () => console.log("set state"));
        this.setState((state: ClickableImageState, props) => {
            console.log("setting state");
            return {...state, selected: !!!state.selected};
        }, () => console.log("fucking updatec"));
        //this.onSelect(this.src);
    }

    render() {         
        if (this.state.selected) {
            return <MDBView zoom key={this.src}>
                <img src={this.src} onClick={async () => await this.setState({selected: true}, () => console.log("set state"))}></img>
            </MDBView>
        } else {
            return <MDBView zoom key={this.src}>
                <img src={this.src}></img>
                <a href="#!" onClick={() => this.select()}><MDBMask overlay="white-strong" className="flex-center"></MDBMask></a>
            </MDBView>
        }
    }
}

export class ImageSelector extends Component {
    images: {[key: string]: ClickableImage}
    state: {selectedSrc: string}

    constructor(props: {images: string[]}) {
        super(props);
        this.images = {};
        props.images.forEach((src: string) => this.images[src] = new ClickableImage({
                src: src, 
                onSelect: (src) => this.handleSelection(src)
            }));
        this.state = {selectedSrc: props.images[0]}
    }

    handleSelection(src: string) {
        const oldSrc: string = this.state.selectedSrc;
        this.setState({selectedSrc: src});
        this.images[oldSrc].setState({selected: false});
        this.images[src].setState({selected: true});
    }

    componentDidMount() {
        //const toSelect: string = Object.keys(this.images)[0];
        //this.images[toSelect].select();
    }

    render() {
        return (
            <MDBContainer>
                <MDBBtnGroup>
                    {Object.keys(this.images).map((src: string) => {
                        return this.images[src].render();
                    })}
                </MDBBtnGroup>
            </MDBContainer>
        );
    }
}