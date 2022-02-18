import { Tensor3D, Tensor4D } from "@tensorflow/tfjs";
import { GridCA } from "./ca-grid";
import { MultiImgCAModel } from "./multi-img-ca";
import * as tf from "@tensorflow/tfjs";
import { Application, RenderTexture, Sprite, Container, Texture } from "pixi.js";

/**
 * A Grid CA using a multi-image CA model
 */
export class MultiImgCAGrid extends GridCA {
    
    model: MultiImgCAModel;
    board: Tensor4D = this.seedMiddlePixelBoard();
    image: Tensor4D;

    /**
     * @inheritdoc
     */
    constructor(model: MultiImgCAModel, imageElement: HTMLImageElement) {
        super();
        this.model = model;
        // load image from element and package it as single image batch (added dim)
        this.image = MultiImgCAGrid.loadImage(imageElement).expandDims(0);
    }

    /**
     * @inheritdoc
     */
    async init(): Promise<MultiImgCAGrid> {
        // TODO: this works but defeats the purpose of async initialization if we have to await it somewhere else
        return Promise.resolve(this);
    }

    /**
     * @inheritdoc
     */
    step(): void {
        // take 1 step with model and update our board
        this.board = this.model.call(this.board, this.image);
    }

    /**
     * @inheritdoc
     */
    rgba(): Float32Array {
        return this.board.slice([0, 0, 0, 0], [1, this.board.shape[1], this.board.shape[2], 4]).squeeze([0]).dataSync() as Float32Array;
    }


    /**
     * 
     */
    seedGrid() {
        this.board = this.seedMiddlePixelBoard();
    }

    /**
     * Load a Tensor representation of an image from React Element 
     */
    static loadImage(image: HTMLImageElement): Tensor3D {
        return tf.browser.fromPixels(image, 4).div(255.0).pad([[16, 16], [16, 16], [0, 0]]) as Tensor3D;
    }

    /**
     * Return a board with a single activated pixel in the center
     */
    private seedMiddlePixelBoard(): Tensor4D {
        // create a 1*72*72*4 (r,g,b,a) grid, with one active pixel in the middle
        const board = tf.buffer([1, 72, 72, 4], 'float32');
        const shape: number[] = board.shape;
        const midX: number = Math.floor(shape[1] / 2);
        const midY: number = Math.floor(shape[2] / 2);
        // activate 1 middle pixel
        board.set(1, 0, midX, midY, 3);
        return tf.concat([board.toTensor(), 
            tf.randomUniform([1,72,72,12])
        ], 3) as Tensor4D;
    }

    /**
     * Given (r, g, b) values where each is the color channel in [0, 256), return the single number representation
     * of that color (hexadecimal equivalent)
     * @param r red channel
     * @param g green channel
     * @param b blue channel
     */
    private rbgToHexNumber(r: number, g: number, b: number) {
        return (65536 * r) + (256 * g) + b;
    }

}

/**
 * 
 * @param image 
 */
export async function createGrid(model: MultiImgCAModel, image: HTMLImageElement): Promise<Application> {
    const caGrid: MultiImgCAGrid = await new MultiImgCAGrid(model, image).init();
    const width = 400;
    const height = 400;
    const app = new Application({
        backgroundColor: 0xffffff,
        width: width,
        height: height
    });
    let renderTexture = RenderTexture.create({
        width: app.screen.width,
        height: app.screen.height
    });

    // create a new sprite that uses the render texture we created above
    const outputSprite = new Sprite(renderTexture);

    // align the sprite
    outputSprite.x = width / 2;
    outputSprite.y = height / 2;
    outputSprite.anchor.set(0.5);

    // this scales the board to the frame size, which will pixelate (good in this case!)
    outputSprite.height = height;
    outputSprite.width = width;

    // add to stage
    app.stage.addChild(outputSprite);

    const stuffContainer = new Container();

    stuffContainer.x = width / 2;
    stuffContainer.y = height / 2;

    app.stage.addChild(stuffContainer);

    app.ticker.maxFPS = 5; 
    app.ticker.add(() => {
        outputSprite.texture = Texture.fromBuffer(caGrid.rgba() as Float32Array, 72, 72);
        caGrid.step();
        //Texture.fromURL("images/emoji_u1f36d.png").then(t => outputSprite.texture = t);
        app.renderer.render(app.stage, renderTexture, false);
    });
    return app;
}
