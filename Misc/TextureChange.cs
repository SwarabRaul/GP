// 21BAI1225 - Swarab Raul

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TextureChange : MonoBehaviour
{
    public Texture closedTexture;
    public Texture openTexture;
    public Renderer objectRenderer;
    public Button changeTextureButton;

    private bool isOpen = false;

    // Start is called before the first frame update
    void Start()
    {
        if(objectRenderer == null) {
            objectRenderer = GetComponent<Renderer>();
        }

        objectRenderer.material.mainTexture = closedTexture;
        changeTextureButton.onClick.AddListener(ChangeTexture);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void ChangeTexture() {
        if(isOpen) {
            objectRenderer.material.mainTexture = closedTexture;
            isOpen = false;
        } else {
            objectRenderer.material.mainTexture = openTexture;
            isOpen = true;
        }
    }
}
