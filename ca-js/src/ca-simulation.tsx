import { Application, RenderTexture, Sprite, Texture } from "pixi.js";
import { Component } from "react";
import { GridCA, GridCAViewOptions } from "./ca-grid";

export interface CASimulationProps {
    viewOpts: Partial<GridCAViewOptions>, 
    gridCA: GridCA,
    id: string
}
const defaultViewOpts: GridCAViewOptions = {
    maxFPS: 5,
    width: 400,
    height: 400,
    maxSteps: undefined
};

/**
 * A component for drawing a CA simulation. Takes a GridCA and runs it in a 
 * PIXI application according to view settings defined in properties
 */
export class CASimulation extends Component {
    mount: any;
    gridCA: GridCA;
    view: GridCAViewOptions;
    steps: number;
    app: Application;
    id: string;
    
    /**
     * A simulation provides a PIXI application displaying the running CA
     * 
     * @param props 
     */
    constructor(props: CASimulationProps) {
        super(props);
        this.view = {
            ...defaultViewOpts,
            ...props.viewOpts
        };
        this.id = props.id;
        this.gridCA = props.gridCA;
        this.steps = 0;
        this.app = this.initApp();
    }

    /**
     * Return a PIXI application that can be added to the DOM in 
     * componentDidMount(); 
     */
    initApp(): Application {
        // create containing application
        const app = new Application({
            backgroundColor: 0xffffff,
            width: this.view.width,
            height: this.view.height
        });
        
        let renderTexture = RenderTexture.create({
            width: app.screen.width,
            height: app.screen.height
        });

        // create a new sprite that uses the render texture we created above
        const outputSprite = new Sprite(renderTexture);

        // align the sprite
        outputSprite.x = this.view.width / 2;
        outputSprite.y = this.view.height / 2;
        outputSprite.anchor.set(0.5);

        // this scales the board to the frame size, which will pixelate (good in this case!)
        outputSprite.height = this.view.height;
        outputSprite.width = this.view.width;

        // add to stage
        app.stage.addChild(outputSprite);

        app.ticker.maxFPS = this.view.maxFPS; 
        
        // add a ticker to do state
        app.ticker.add(() => {
            if (!this.view.maxSteps || this.steps <= this.view.maxSteps) {
                outputSprite.texture = Texture.fromBuffer(this.gridCA.rgba() as Float32Array, 72, 72);
                this.gridCA.step();
                app.renderer.render(app.stage, renderTexture, false);
            }
        });

        return app;
    }
    
    async componentDidMount() {
        document.getElementById(this.id)?.appendChild(this.app.view);
    }

    render() {
        // return app to display
        return <div id={this.id}></div>;
    }
}


